from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..codes import load_code_artifact
from ..config import ExperimentConfig
from ..decoders.features import build_decoder_features
from ..models.posterior_scorer import PosteriorSchedulerNet
from ..utils.env import configure_torch_threads, set_global_seed
from ..utils.io import write_dataframe_csv, write_json
from .data_generation import load_dataset_arrays


def _prepare_supervised_arrays(
    arrays: Dict[str, np.ndarray],
    code,
    num_segments: int,
    max_weight_class: int,
    confidence_weight_limit: int,
    confidence_rank_limit: int,
) -> Dict[str, np.ndarray]:
    bit_features = []
    global_features = []
    bit_targets = []
    segment_targets = []
    weight_targets = []
    confidence_targets = []
    for idx in range(arrays["llr"].shape[0]):
        feat = build_decoder_features(
            llr=arrays["llr"][idx],
            hard_bits=arrays["hard_bits"][idx],
            code=code,
            snr_db=float(arrays["snr_db"][idx]),
            profile_id=int(arrays["profile_id"][idx]),
            num_segments=num_segments,
            error_mask=arrays["error_mask"][idx],
            confidence_weight_threshold=confidence_weight_limit,
            confidence_rank_limit=confidence_rank_limit,
        )
        bit_features.append(feat.bit_features)
        global_features.append(feat.global_features)
        bit_targets.append(feat.error_mask_ranked.astype(np.float32))
        segment_targets.append(feat.segment_labels.astype(np.float32))
        weight_targets.append(min(int(feat.weight_label), max_weight_class + 1))
        confidence_targets.append(float(feat.confidence_label))
    return {
        "bit_features": np.stack(bit_features).astype(np.float32),
        "global_features": np.stack(global_features).astype(np.float32),
        "bit_targets": np.stack(bit_targets).astype(np.float32),
        "segment_targets": np.stack(segment_targets).astype(np.float32),
        "weight_targets": np.asarray(weight_targets, dtype=np.int64),
        "confidence_targets": np.asarray(confidence_targets, dtype=np.float32),
    }


def _build_model(cfg: ExperimentConfig, n_bits: int, per_bit_dim: int, global_dim: int) -> PosteriorSchedulerNet:
    return PosteriorSchedulerNet(
        n_bits=n_bits,
        per_bit_dim=per_bit_dim,
        global_dim=global_dim,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        ff_multiplier=cfg.model.ff_multiplier,
        dropout=cfg.model.dropout,
        num_segments=cfg.model.num_segments,
        max_weight_class=cfg.model.max_weight_class,
    )


def _classification_metrics(outputs, bit_y, segment_y, weight_y, conf_y) -> Dict[str, float]:
    bit_prob = torch.sigmoid(outputs["bit_logits"])
    segment_prob = torch.sigmoid(outputs["segment_logits"])
    weight_pred = torch.argmax(outputs["weight_logits"], dim=-1)
    conf_prob = torch.sigmoid(outputs["confidence_logit"])

    bit_pred = (bit_prob >= 0.5).float()
    bit_tp = float(((bit_pred == 1.0) & (bit_y == 1.0)).sum().item())
    bit_fp = float(((bit_pred == 1.0) & (bit_y == 0.0)).sum().item())
    bit_fn = float(((bit_pred == 0.0) & (bit_y == 1.0)).sum().item())
    bit_precision = bit_tp / max(1.0, bit_tp + bit_fp)
    bit_recall = bit_tp / max(1.0, bit_tp + bit_fn)

    seg_pred = (segment_prob >= 0.5).float()
    seg_acc = float((seg_pred == segment_y).float().mean().item())
    weight_acc = float((weight_pred == weight_y).float().mean().item())
    overflow_true = (weight_y == weight_y.max()).float()
    overflow_pred = (weight_pred == weight_y.max()).float()
    overflow_tp = float(((overflow_true == 1.0) & (overflow_pred == 1.0)).sum().item())
    overflow_fn = float(((overflow_true == 1.0) & (overflow_pred == 0.0)).sum().item())
    overflow_recall = overflow_tp / max(1.0, overflow_tp + overflow_fn)
    conf_acc = float((((conf_prob >= 0.5).float()) == conf_y).float().mean().item())
    conf_brier = float(torch.mean((conf_prob - conf_y) ** 2).item())
    return {
        "bit_precision": bit_precision,
        "bit_recall": bit_recall,
        "segment_acc": seg_acc,
        "weight_acc": weight_acc,
        "overflow_recall": overflow_recall,
        "confidence_acc": conf_acc,
        "confidence_brier": conf_brier,
    }


def train_model(cfg: ExperimentConfig, output_dir: Path, logger) -> Dict[str, object]:
    set_global_seed(cfg.seed)
    configure_torch_threads(cfg.resources.train_torch_threads, cfg.resources.train_torch_interop_threads)
    device = torch.device("cpu")

    code = load_code_artifact(output_dir / "artifacts" / "code_artifact.json")
    train_arrays = load_dataset_arrays(output_dir / "datasets" / "train")
    val_arrays = load_dataset_arrays(output_dir / "datasets" / "val")

    logger.info("Preparing supervised tensors from train/val datasets")
    train_data = _prepare_supervised_arrays(
        train_arrays,
        code,
        cfg.model.num_segments,
        cfg.model.max_weight_class,
        cfg.train.confidence_weight_limit,
        cfg.train.confidence_rank_limit,
    )
    val_data = _prepare_supervised_arrays(
        val_arrays,
        code,
        cfg.model.num_segments,
        cfg.model.max_weight_class,
        cfg.train.confidence_weight_limit,
        cfg.train.confidence_rank_limit,
    )

    train_ds = TensorDataset(
        torch.from_numpy(train_data["bit_features"]),
        torch.from_numpy(train_data["global_features"]),
        torch.from_numpy(train_data["bit_targets"]),
        torch.from_numpy(train_data["segment_targets"]),
        torch.from_numpy(train_data["weight_targets"]),
        torch.from_numpy(train_data["confidence_targets"]),
    )
    val_ds = TensorDataset(
        torch.from_numpy(val_data["bit_features"]),
        torch.from_numpy(val_data["global_features"]),
        torch.from_numpy(val_data["bit_targets"]),
        torch.from_numpy(val_data["segment_targets"]),
        torch.from_numpy(val_data["weight_targets"]),
        torch.from_numpy(val_data["confidence_targets"]),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.resources.train_loader_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=max(0, cfg.resources.train_loader_workers // 2),
        drop_last=False,
    )

    model = _build_model(
        cfg,
        n_bits=code.n,
        per_bit_dim=train_data["bit_features"].shape[-1],
        global_dim=train_data["global_features"].shape[-1],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    bit_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.train.positive_bit_class_weight, dtype=torch.float32))
    segment_loss_fn = nn.BCEWithLogitsLoss()
    weight_loss_fn = nn.CrossEntropyLoss()
    confidence_loss_fn = nn.BCEWithLogitsLoss()

    history = []
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        train_totals = {"loss": 0.0, "bit": 0.0, "segment": 0.0, "weight": 0.0, "confidence": 0.0}
        train_metric_totals = {k: 0.0 for k in ["bit_precision", "bit_recall", "segment_acc", "weight_acc", "overflow_recall", "confidence_acc", "confidence_brier"]}
        train_samples = 0
        for batch in train_loader:
            bit_x, glob_x, bit_y, segment_y, weight_y, conf_y = [tensor.to(device) for tensor in batch]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(bit_x, glob_x)
            bit_loss = bit_loss_fn(outputs["bit_logits"], bit_y)
            segment_loss = segment_loss_fn(outputs["segment_logits"], segment_y)
            weight_loss = weight_loss_fn(outputs["weight_logits"], weight_y)
            confidence_loss = confidence_loss_fn(outputs["confidence_logit"], conf_y)
            total_loss = (
                cfg.train.bit_loss_weight * bit_loss
                + cfg.train.segment_loss_weight * segment_loss
                + cfg.train.weight_loss_weight * weight_loss
                + cfg.train.confidence_loss_weight * confidence_loss
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
            optimizer.step()

            batch_size = int(bit_x.size(0))
            train_samples += batch_size
            train_totals["loss"] += float(total_loss.item()) * batch_size
            train_totals["bit"] += float(bit_loss.item()) * batch_size
            train_totals["segment"] += float(segment_loss.item()) * batch_size
            train_totals["weight"] += float(weight_loss.item()) * batch_size
            train_totals["confidence"] += float(confidence_loss.item()) * batch_size
            batch_metrics = _classification_metrics(outputs, bit_y, segment_y, weight_y, conf_y)
            for key, value in batch_metrics.items():
                train_metric_totals[key] += float(value) * batch_size

        model.eval()
        val_totals = {"loss": 0.0, "bit": 0.0, "segment": 0.0, "weight": 0.0, "confidence": 0.0}
        val_metric_totals = {k: 0.0 for k in ["bit_precision", "bit_recall", "segment_acc", "weight_acc", "overflow_recall", "confidence_acc", "confidence_brier"]}
        val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                bit_x, glob_x, bit_y, segment_y, weight_y, conf_y = [tensor.to(device) for tensor in batch]
                outputs = model(bit_x, glob_x)
                bit_loss = bit_loss_fn(outputs["bit_logits"], bit_y)
                segment_loss = segment_loss_fn(outputs["segment_logits"], segment_y)
                weight_loss = weight_loss_fn(outputs["weight_logits"], weight_y)
                confidence_loss = confidence_loss_fn(outputs["confidence_logit"], conf_y)
                total_loss = (
                    cfg.train.bit_loss_weight * bit_loss
                    + cfg.train.segment_loss_weight * segment_loss
                    + cfg.train.weight_loss_weight * weight_loss
                    + cfg.train.confidence_loss_weight * confidence_loss
                )
                batch_size = int(bit_x.size(0))
                val_samples += batch_size
                val_totals["loss"] += float(total_loss.item()) * batch_size
                val_totals["bit"] += float(bit_loss.item()) * batch_size
                val_totals["segment"] += float(segment_loss.item()) * batch_size
                val_totals["weight"] += float(weight_loss.item()) * batch_size
                val_totals["confidence"] += float(confidence_loss.item()) * batch_size
                batch_metrics = _classification_metrics(outputs, bit_y, segment_y, weight_y, conf_y)
                for key, value in batch_metrics.items():
                    val_metric_totals[key] += float(value) * batch_size

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_totals["loss"] / max(1, train_samples),
            "train_bit_loss": train_totals["bit"] / max(1, train_samples),
            "train_segment_loss": train_totals["segment"] / max(1, train_samples),
            "train_weight_loss": train_totals["weight"] / max(1, train_samples),
            "train_confidence_loss": train_totals["confidence"] / max(1, train_samples),
            "val_loss": val_totals["loss"] / max(1, val_samples),
            "val_bit_loss": val_totals["bit"] / max(1, val_samples),
            "val_segment_loss": val_totals["segment"] / max(1, val_samples),
            "val_weight_loss": val_totals["weight"] / max(1, val_samples),
            "val_confidence_loss": val_totals["confidence"] / max(1, val_samples),
        }
        for key, total in train_metric_totals.items():
            epoch_record[f"train_{key}"] = total / max(1, train_samples)
        for key, total in val_metric_totals.items():
            epoch_record[f"val_{key}"] = total / max(1, val_samples)
        history.append(epoch_record)
        logger.info(
            "Epoch %d | train_loss=%.4f | val_loss=%.4f | val_weight_acc=%.4f | val_conf_brier=%.4f",
            epoch,
            epoch_record["train_loss"],
            epoch_record["val_loss"],
            epoch_record["val_weight_acc"],
            epoch_record["val_confidence_brier"],
        )

        if epoch_record["val_loss"] < best_val_loss:
            best_val_loss = epoch_record["val_loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.train.early_stopping_patience:
                logger.info("Early stopping triggered after epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / "posterior_scorer.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.to_dict(),
            "n_bits": code.n,
            "per_bit_dim": train_data["bit_features"].shape[-1],
            "global_dim": train_data["global_features"].shape[-1],
        },
        checkpoint_path,
    )

    history_df = pd.DataFrame(history)
    write_dataframe_csv(history_df, output_dir / "training" / "training_history.csv")
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "num_train_samples": int(train_data["bit_features"].shape[0]),
        "num_val_samples": int(val_data["bit_features"].shape[0]),
        "epochs_ran": int(len(history)),
        "confidence_rank_limit": int(cfg.train.confidence_rank_limit),
        "confidence_weight_limit": int(cfg.train.confidence_weight_limit),
    }
    if history:
        best_row = min(history, key=lambda row: row["val_loss"])
        summary.update(
            {
                "best_val_weight_acc": float(best_row["val_weight_acc"]),
                "best_val_confidence_brier": float(best_row["val_confidence_brier"]),
                "best_val_overflow_recall": float(best_row["val_overflow_recall"]),
            }
        )
    write_json(summary, output_dir / "training" / "training_summary.json")
    return summary


def load_trained_model(output_dir: Path, device: str = "cpu") -> PosteriorSchedulerNet:
    checkpoint = torch.load(output_dir / "checkpoints" / "posterior_scorer.pt", map_location=device)
    cfg = ExperimentConfig()
    if isinstance(checkpoint.get("config"), dict):
        model_cfg = checkpoint["config"].get("model", {})
        for key, value in model_cfg.items():
            if hasattr(cfg.model, key):
                setattr(cfg.model, key, value)
    model = _build_model(
        cfg,
        n_bits=int(checkpoint["n_bits"]),
        per_bit_dim=int(checkpoint["per_bit_dim"]),
        global_dim=int(checkpoint["global_dim"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model
