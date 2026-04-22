from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..codes.factory import build_code
from ..models.rescue_net import RescueNet
from ..training.dataset import NPZShardDataset, infer_shapes
from ..utils.io import ensure_dir, write_json
from ..utils.logging import get_logger


def _bit_rank_loss(bit_logits: torch.Tensor, bit_labels: torch.Tensor) -> torch.Tensor:
    # Encourage positive target bits to rank above negatives without all-pairs cost.
    pos = bit_labels > 0.5
    neg = ~pos
    losses = []
    for b in range(bit_logits.shape[0]):
        if pos[b].any() and neg[b].any():
            p = bit_logits[b][pos[b]].mean()
            # Hard negative average: top 16 negative logits.
            neg_logits = bit_logits[b][neg[b]]
            k = min(16, neg_logits.numel())
            n = torch.topk(neg_logits, k=k).values.mean()
            losses.append(torch.relu(1.0 - p + n))
    if not losses:
        return bit_logits.new_tensor(0.0)
    return torch.stack(losses).mean()


class MultiTaskLoss(nn.Module):
    def __init__(self, cfg: Dict[str, object]):
        super().__init__()
        lw = cfg["train"]["loss_weights"]
        self.loss_weights = {k: float(v) for k, v in lw.items()}
        bit_pos_w = float(cfg["train"].get("bit_pos_weight", 4.0))
        reach_pos_w = float(cfg["train"].get("reach_pos_weight", 4.0))
        self.bit_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(bit_pos_w))
        self.seg_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(reach_pos_w))
        self.cls_ce = nn.CrossEntropyLoss()
        self.bin_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(reach_pos_w))

    def forward(self, model: RescueNet, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses = {}
        losses["bit"] = self.bit_bce(out["bit_logits"], batch["bit_labels"])
        losses["segment"] = self.seg_bce(out["segment_logits"], batch["segment_labels"])
        losses["weight"] = self.cls_ce(out["weight_logits"], batch["weight_label"].long().view(-1))
        losses["rank"] = _bit_rank_loss(out["bit_logits"], batch["bit_labels"])
        losses["standard_reachable"] = self.bin_bce(out["standard_logits"], batch["standard_reachable"].view(-1))
        losses["expanded_reachable"] = self.bin_bce(out["expanded_logits"], batch["expanded_reachable"].view(-1))
        losses["rescueable"] = self.bin_bce(out["rescue_logits"], batch["rescueable"].view(-1))

        # Candidate rerank loss only when there is a positive candidate.
        cand_labels = batch.get("candidate_labels")
        cand_valid = batch.get("candidate_valid")
        if cand_labels is not None and cand_valid is not None and "candidate_features" in batch:
            positive_exists = (cand_labels * cand_valid).sum(dim=1) > 0
            if positive_exists.any():
                scores = model.reranker(out["packet_embedding"], batch["candidate_features"])
                masked_scores = scores.masked_fill(cand_valid <= 0, -1e9)
                target_idx = torch.argmax(cand_labels, dim=1)
                losses["rerank"] = nn.CrossEntropyLoss()(masked_scores[positive_exists], target_idx[positive_exists].long())
            else:
                losses["rerank"] = out["bit_logits"].new_tensor(0.0)
        else:
            losses["rerank"] = out["bit_logits"].new_tensor(0.0)

        total = out["bit_logits"].new_tensor(0.0)
        metrics = {}
        for k, v in losses.items():
            total = total + self.loss_weights.get(k, 0.0) * v
            metrics[k] = float(v.detach().cpu())
        metrics["total"] = float(total.detach().cpu())
        return total, metrics


def _move(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def train_model(cfg: Dict[str, object]) -> None:
    logger = get_logger("train")
    out_dir = ensure_dir(Path(cfg["project"]["output_dir"]))
    train_dir = out_dir / "datasets" / "train"
    val_dir = out_dir / "datasets" / "val"
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    hist_path = ensure_dir(out_dir / "training") / "training_history.csv"
    write_json(cfg, out_dir / "artifacts" / "resolved_config.json")

    shapes = infer_shapes(train_dir)
    code = build_code(cfg["code"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s with shapes=%s", device, shapes)

    model = RescueNet(
        num_var_features=shapes["num_var_features"],
        num_check_features=shapes["num_check_features"],
        num_global_features=shapes["num_global_features"],
        n=shapes["n"],
        m=shapes["m"],
        num_segments=int(cfg["model"]["num_segments"]),
        max_weight_class=int(cfg["model"]["max_weight_class"]),
        hidden_dim=int(cfg["model"]["graph_hidden_dim"]),
        graph_layers=int(cfg["model"]["graph_layers"]),
        top_k_tokens=int(cfg["model"]["top_k_tokens"]),
        transformer_heads=int(cfg["model"].get("transformer_heads", 4)),
        transformer_layers=int(cfg["model"].get("transformer_layers", 1)),
        dropout=float(cfg["train"].get("dropout", 0.05)),
        candidate_feature_dim=shapes.get("candidate_feature_dim", 8),
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"].get("weight_decay", 1e-5)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, int(cfg["train"]["epochs"])))
    start_epoch = 1
    latest = ckpt_dir / "rescue_net_latest.pt"
    if bool(cfg["train"].get("resume", True)) and latest.exists():
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optim.load_state_dict(ckpt["optim_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        logger.info("Resumed from %s at epoch %d", latest, start_epoch)

    train_ds = NPZShardDataset(train_dir)
    val_ds = NPZShardDataset(val_dir)
    train_loader = DataLoader(train_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=0)

    criterion = MultiTaskLoss(cfg)
    h_dense = torch.tensor(code.h.astype(np.float32), dtype=torch.float32, device=device)
    deg_v = torch.tensor(np.maximum(code.deg_v.astype(np.float32), 1.0), dtype=torch.float32, device=device)
    deg_c = torch.tensor(np.maximum(code.deg_c.astype(np.float32), 1.0), dtype=torch.float32, device=device)

    exists = hist_path.exists()
    with hist_path.open("a", newline="", encoding="utf-8") as f:
        fieldnames = ["epoch", "train_loss", "val_loss", "val_weight_acc", "val_rescue_acc", "val_std_brier", "lr", "train_samples_seen", "val_samples_seen"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()

        for epoch in range(start_epoch, int(cfg["train"]["epochs"]) + 1):
            model.train()
            train_losses = []
            for batch in train_loader:
                batch = _move(batch, device)
                optim.zero_grad(set_to_none=True)
                out = model(batch["var_features"], batch["check_features"], batch["global_features"], batch["heuristic_order"], h_dense, deg_v, deg_c)
                loss, _ = criterion(model, out, batch)
                loss.backward()
                if float(cfg["train"].get("grad_clip", 0.0)) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
                optim.step()
                train_losses.append(float(loss.detach().cpu()))
            scheduler.step()

            model.eval()
            val_losses = []
            weight_accs = []
            rescue_accs = []
            briers = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = _move(batch, device)
                    out = model(batch["var_features"], batch["check_features"], batch["global_features"], batch["heuristic_order"], h_dense, deg_v, deg_c)
                    loss, _ = criterion(model, out, batch)
                    val_losses.append(float(loss.detach().cpu()))
                    pred_w = out["weight_logits"].argmax(dim=-1)
                    weight_accs.append(float((pred_w == batch["weight_label"].long().view(-1)).float().mean().cpu()))
                    rescue_pred = (torch.sigmoid(out["rescue_logits"]) > 0.5).float()
                    rescue_accs.append(float((rescue_pred == batch["rescueable"].view(-1)).float().mean().cpu()))
                    std_p = torch.sigmoid(out["standard_logits"])
                    briers.append(float(torch.mean((std_p - batch["standard_reachable"].view(-1)) ** 2).cpu()))

            record = {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                "val_loss": float(np.mean(val_losses)) if val_losses else float("nan"),
                "val_weight_acc": float(np.mean(weight_accs)) if weight_accs else float("nan"),
                "val_rescue_acc": float(np.mean(rescue_accs)) if rescue_accs else float("nan"),
                "val_std_brier": float(np.mean(briers)) if briers else float("nan"),
                "lr": float(scheduler.get_last_lr()[0]),
                "train_samples_seen": len(train_ds),
                "val_samples_seen": len(val_ds),
            }
            writer.writerow(record)
            f.flush()
            logger.info("Epoch %d | train_loss=%.4f val_loss=%.4f val_weight_acc=%.4f val_rescue_acc=%.4f",
                        epoch, record["train_loss"], record["val_loss"], record["val_weight_acc"], record["val_rescue_acc"])

            ckpt = {
                "epoch": epoch,
                "cfg": cfg,
                "shapes": shapes,
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }
            torch.save(ckpt, latest)
            torch.save(ckpt, ckpt_dir / "rescue_net.pt")

    write_json({"epochs": int(cfg["train"]["epochs"]), "checkpoint": str(ckpt_dir / "rescue_net.pt")}, out_dir / "training" / "training_summary.json")
