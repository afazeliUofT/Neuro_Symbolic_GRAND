from __future__ import annotations

from pathlib import Path
from typing import Dict
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..codes.factory import build_code
from ..models.rescue_net import RescueNet
from .dataset import SupervisedShardDataset
from ..utils.io import ensure_dir, save_json


def _bit_rank_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    pos_mask = labels > 0.5
    neg_mask = labels <= 0.5
    valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    pos_scores = logits.masked_fill(~pos_mask, -1e9).max(dim=1).values
    neg_scores = logits.masked_fill(~neg_mask, -1e9).max(dim=1).values
    return torch.relu(margin - (pos_scores[valid] - neg_scores[valid])).mean()


class MultiTaskLoss(nn.Module):
    def __init__(self, cfg: Dict[str, object]):
        super().__init__()
        lw = cfg["train"]["loss_weights"]
        self.loss_weights = {k: float(v) for k, v in lw.items()}
        pos_w = float(cfg["train"].get("bit_pos_weight", 10.0))
        self.bit_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w))
        self.seg_bce = nn.BCEWithLogitsLoss()
        self.cls_ce = nn.CrossEntropyLoss()
        self.bin_bce = nn.BCEWithLogitsLoss()

    def forward(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, float]]:
        losses = {}
        losses["bit"] = self.bit_bce(out["bit_logits"], batch["bit_labels"])
        losses["segment"] = self.seg_bce(out["segment_logits"], batch["segment_labels"])
        losses["weight"] = self.cls_ce(out["weight_logits"], batch["weight_label"])
        losses["rank"] = _bit_rank_loss(out["bit_logits"], batch["bit_labels"])
        losses["standard_reachable"] = self.bin_bce(out["standard_logits"], batch["standard_reachable"])
        losses["expanded_reachable"] = self.bin_bce(out["expanded_logits"], batch["expanded_reachable"])
        losses["rescueable"] = self.bin_bce(out["rescue_logits"], batch["rescueable"])
        cand_scores = out["candidate_scores"]
        cand_labels = batch["candidate_labels"]
        cand_valid = batch["candidate_valid"]
        masked_scores = cand_scores.masked_fill(cand_valid <= 0, -1e9)
        positive_exists = (cand_labels * cand_valid).sum(dim=1) > 0
        if positive_exists.any():
            target_idx = torch.argmax(cand_labels, dim=1)
            losses["rerank"] = nn.functional.cross_entropy(masked_scores[positive_exists], target_idx[positive_exists])
        else:
            losses["rerank"] = cand_scores.sum() * 0.0
        total = 0.0
        for name, val in losses.items():
            total = total + self.loss_weights.get(name, 0.0) * val
        return total, {k: float(v.detach().cpu()) for k, v in losses.items()}


def _accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) > 0.5).float()
    return float((pred == target).float().mean().detach().cpu())


def _brier(logits: torch.Tensor, target: torch.Tensor) -> float:
    prob = torch.sigmoid(logits)
    return float(torch.mean((prob - target) ** 2).detach().cpu())


def train_rescue_model(cfg: Dict[str, object], output_dir: Path, logger) -> None:
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    best_path = ckpt_dir / "rescue_net.pt"
    latest_path = ckpt_dir / "rescue_net_latest.pt"
    summary_path = output_dir / "training" / "training_summary.json"
    if summary_path.exists() and best_path.exists():
        logger.info("Training already complete; found %s and %s. Skipping.", summary_path, best_path)
        return

    train_ds = SupervisedShardDataset(output_dir, "train")
    val_ds = SupervisedShardDataset(output_dir, "val")
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
    )

    code = build_code(cfg["code"])
    device = torch.device(str(cfg["train"].get("device", "cpu")))
    torch.set_num_threads(int(cfg["train"].get("torch_threads", 32)))
    model = RescueNet(
        num_var_features=int(train_ds.data["var_features"].shape[-1]),
        num_check_features=int(train_ds.data["check_features"].shape[-1]),
        num_global_features=int(train_ds.data["global_features"].shape[-1]),
        n=code.n,
        m=code.m,
        num_segments=int(cfg["model"]["num_segments"]),
        max_weight_class=int(cfg["model"]["max_weight_class"]),
        hidden_dim=int(cfg["model"]["graph_hidden_dim"]),
        graph_layers=int(cfg["model"]["graph_layers"]),
        top_k_tokens=int(cfg["model"]["top_k_tokens"]),
        transformer_heads=int(cfg["model"]["transformer_heads"]),
        transformer_layers=int(cfg["model"]["transformer_layers"]),
        dropout=float(cfg["train"]["dropout"]),
        candidate_feature_dim=int(train_ds.data["candidate_features"].shape[-1]),
    ).to(device)
    h_dense = torch.tensor(code.h.astype(np.float32), dtype=torch.float32, device=device)
    deg_v = torch.tensor(np.maximum(code.deg_v.astype(np.float32), 1.0), dtype=torch.float32, device=device)
    deg_c = torch.tensor(np.maximum(code.deg_c.astype(np.float32), 1.0), dtype=torch.float32, device=device)

    criterion = MultiTaskLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"].get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg["train"].get("lr_factor", 0.5)),
        patience=int(cfg["train"].get("lr_patience", 3)),
        min_lr=float(cfg["train"].get("min_lr", 1e-4)),
    )

    best = {"val_loss": float("inf"), "epoch": -1}
    history = []
    patience = int(cfg["train"].get("early_stopping_patience", 6))
    stale = 0
    start_epoch = 1
    total_epochs = int(cfg["train"]["epochs"])

    if latest_path.exists():
        state = torch.load(latest_path, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        best = state.get("best", best)
        history = state.get("history", history)
        stale = int(state.get("stale", stale))
        start_epoch = int(state.get("epoch", 0)) + 1
        logger.info("Resuming training from epoch %d/%d", start_epoch, total_epochs)

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(batch["var_features"], batch["check_features"], batch["global_features"],
                        batch["heuristic_order"], h_dense, deg_v, deg_c, batch["candidate_features"])
            loss, _ = criterion(out, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        metric_accum = {"weight_acc": [], "std_brier": [], "rescue_acc": []}
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch["var_features"], batch["check_features"], batch["global_features"],
                            batch["heuristic_order"], h_dense, deg_v, deg_c, batch["candidate_features"])
                loss, _ = criterion(out, batch)
                val_losses.append(float(loss.detach().cpu()))
                metric_accum["weight_acc"].append(float((torch.argmax(out["weight_logits"], dim=1) == batch["weight_label"]).float().mean().cpu()))
                metric_accum["std_brier"].append(_brier(out["standard_logits"], batch["standard_reachable"]))
                metric_accum["rescue_acc"].append(_accuracy(out["rescue_logits"], batch["rescueable"]))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)
        lr = float(optimizer.param_groups[0]["lr"])
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_weight_acc": float(np.mean(metric_accum["weight_acc"])),
            "val_std_brier": float(np.mean(metric_accum["std_brier"])),
            "val_rescue_acc": float(np.mean(metric_accum["rescue_acc"])),
            "lr": lr,
            "train_samples_seen": len(train_ds),
            "val_samples_seen": len(val_ds),
        }
        history.append(record)
        logger.info(
            "Epoch %d | train_loss=%.4f | val_loss=%.4f | val_weight_acc=%.4f | val_rescue_acc=%.4f | val_std_brier=%.4f | lr=%.5f",
            epoch, train_loss, val_loss, record["val_weight_acc"], record["val_rescue_acc"], record["val_std_brier"], lr,
        )

        if val_loss < best["val_loss"]:
            best.update({"val_loss": val_loss, "epoch": epoch})
            stale = 0
            torch.save({"model_state": model.state_dict(), "cfg": cfg}, best_path)
        else:
            stale += 1

        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "cfg": cfg,
            "epoch": epoch,
            "best": best,
            "history": history,
            "stale": stale,
        }, latest_path)

        ensure_dir(output_dir / "training")
        import pandas as pd
        pd.DataFrame(history).to_csv(output_dir / "training" / "training_history_partial.csv", index=False)

        if stale >= patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    history_path = output_dir / "training" / "training_history.csv"
    ensure_dir(history_path.parent)
    import pandas as pd
    pd.DataFrame(history).to_csv(history_path, index=False)
    summary = {
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "epochs_completed": len(history),
        "best_epoch": int(best["epoch"]),
        "best_val_loss": float(best["val_loss"]),
        "checkpoint_path": str(best_path),
        "latest_checkpoint_path": str(latest_path),
        "history_path": str(history_path),
    }
    save_json(summary, output_dir / "training" / "training_summary.json")
