from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .code import build_code, write_code_summary
from .config import save_resolved_config
from .model import build_rescue_net


def _load_split(out_dir: Path, split: str) -> Dict[str, np.ndarray]:
    files = sorted((out_dir / "datasets" / split).glob("*.npz"))
    if not files:
        raise RuntimeError(f"No dataset shards found for split={split} under {out_dir/'datasets'/split}")
    arrays: Dict[str, List[np.ndarray]] = {}
    for p in files:
        with np.load(p, allow_pickle=False) as z:
            for k in z.files:
                arrays.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v, axis=0) for k, v in arrays.items()}


def _weighted_bce(tf, logits, labels, pos_weight: float):
    labels = tf.cast(labels, tf.float32)
    logits = tf.cast(logits, tf.float32)
    loss = tf.nn.weighted_cross_entropy_with_logits(labels=labels, logits=logits, pos_weight=float(pos_weight))
    return tf.reduce_mean(loss)


def _candidate_stats(tf, scores, labels, valid):
    scores = tf.cast(scores, tf.float32)
    labels = tf.cast(labels, tf.float32)
    valid = tf.cast(valid, tf.float32)
    masked_scores = scores + (1.0 - valid) * tf.constant(-1e9, tf.float32)
    pos = labels * valid
    any_pos = tf.reduce_sum(pos, axis=1) > 0
    pos_dist = pos / tf.maximum(tf.reduce_sum(pos, axis=1, keepdims=True), 1.0)
    ce = -tf.reduce_sum(pos_dist * tf.nn.log_softmax(masked_scores, axis=1), axis=1)
    ce = tf.where(any_pos, ce, tf.zeros_like(ce))
    loss = tf.reduce_sum(ce) / tf.maximum(tf.reduce_sum(tf.cast(any_pos, tf.float32)), 1.0)
    top = tf.argmax(masked_scores, axis=1, output_type=tf.int32)
    batch = tf.range(tf.shape(scores)[0], dtype=tf.int32)
    top_label = tf.gather_nd(labels, tf.stack([batch, top], axis=1))
    top_acc = tf.reduce_sum(tf.where(any_pos, top_label, tf.zeros_like(top_label))) / tf.maximum(tf.reduce_sum(tf.cast(any_pos, tf.float32)), 1.0)
    pos_rate = tf.reduce_mean(tf.cast(any_pos, tf.float32))
    return loss, top_acc, pos_rate


def _make_dataset(tf, arrs: Dict[str, np.ndarray], batch_size: int, shuffle: bool, seed: int):
    keys = [
        "var_features", "check_features", "global_features", "bit_labels", "segment_labels", "weight_label",
        "standard_reachable", "expanded_reachable", "rescueable", "candidate_features", "candidate_labels", "candidate_valid",
    ]
    data = {k: arrs[k] for k in keys}
    ds = tf.data.Dataset.from_tensor_slices(data)
    if shuffle:
        ds = ds.shuffle(min(len(arrs["weight_label"]), 20000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def train(cfg: Dict[str, Any]) -> None:
    import tensorflow as tf
    tcfg = cfg.get("train", {})
    if bool(tcfg.get("require_gpu", False)) and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("train.require_gpu=true but TensorFlow sees no GPU")
    try:
        tf.config.threading.set_intra_op_parallelism_threads(int(tcfg.get("cpu_threads", 0)) or 0)
        tf.config.threading.set_inter_op_parallelism_threads(max(1, min(8, int(tcfg.get("cpu_threads", 8)))))
    except Exception:
        pass
    mp = tcfg.get("mixed_precision", False)
    if mp:
        policy = str(mp) if isinstance(mp, str) else "mixed_float16"
        if policy == "bfloat16":
            policy = "mixed_bfloat16"
        elif policy == "float16":
            policy = "mixed_float16"
        tf.keras.mixed_precision.set_global_policy(policy)
    print(f"TensorFlow version={tf.__version__} GPUs={tf.config.list_physical_devices('GPU')} mixed_policy={tf.keras.mixed_precision.global_policy()}")
    out_dir = Path(cfg["project"]["output_dir"])
    code = build_code(cfg)
    write_code_summary(code, out_dir)
    save_resolved_config(cfg, out_dir)
    train_arr = _load_split(out_dir, "train")
    val_arr = _load_split(out_dir, "val")
    batch_size = int(tcfg.get("batch_size", 96))
    seed = int(cfg.get("project", {}).get("seed", 31415))
    train_ds = _make_dataset(tf, train_arr, batch_size, True, seed)
    val_ds = _make_dataset(tf, val_arr, batch_size, False, seed)
    model = build_rescue_net(code, cfg)
    # Build variables.
    sample = next(iter(train_ds))
    _ = model(sample, training=False)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights_path = ckpt_dir / "rescue_net_tf.weights.h5"
    best_path = ckpt_dir / "rescue_net_tf.best.weights.h5"
    if bool(tcfg.get("resume", False)) and weights_path.exists():
        model.load_weights(str(weights_path))
        print(f"Resumed weights from {weights_path}")
    lr = float(tcfg.get("lr", 8e-4))
    opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=float(tcfg.get("weight_decay", 1e-5)), clipnorm=float(tcfg.get("grad_clip", 1.0)))
    lw = {k: float(v) for k, v in cfg["train"].get("loss_weights", {}).items()}
    bit_pos_w = float(tcfg.get("bit_pos_weight", 8.0))
    reach_pos_w = float(tcfg.get("reach_pos_weight", 8.0))

    def multitask_loss(batch, out):
        losses = {}
        losses["bit"] = _weighted_bce(tf, out["bit_logits"], batch["bit_labels"], bit_pos_w)
        losses["segment"] = _weighted_bce(tf, out["segment_logits"], batch["segment_labels"], 3.0)
        losses["weight"] = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(batch["weight_label"], out["weight_logits"], from_logits=True))
        losses["standard_reachable"] = _weighted_bce(tf, out["standard_logits"], tf.reshape(batch["standard_reachable"], [-1]), reach_pos_w)
        losses["expanded_reachable"] = _weighted_bce(tf, out["expanded_logits"], tf.reshape(batch["expanded_reachable"], [-1]), reach_pos_w)
        losses["rescueable"] = _weighted_bce(tf, out["rescue_logits"], tf.reshape(batch["rescueable"], [-1]), reach_pos_w)
        rerank_loss, cand_top1, cand_pos = _candidate_stats(tf, out["candidate_scores"], batch["candidate_labels"], batch["candidate_valid"])
        losses["rerank"] = rerank_loss
        losses["rank"] = tf.constant(0.0, dtype=tf.float32)
        total = tf.constant(0.0, dtype=tf.float32)
        for k, v in losses.items():
            total = total + float(lw.get(k, 0.0)) * tf.cast(v, tf.float32)
        return total, losses, cand_top1, cand_pos

    @tf.function(jit_compile=False)
    def train_step(batch):
        with tf.GradientTape() as tape:
            out = model(batch, training=True)
            loss, losses, cand_top1, cand_pos = multitask_loss(batch, out)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss, cand_top1, cand_pos

    @tf.function(jit_compile=False)
    def val_step(batch):
        out = model(batch, training=False)
        loss, losses, cand_top1, cand_pos = multitask_loss(batch, out)
        weight_pred = tf.argmax(out["weight_logits"], axis=1, output_type=tf.int32)
        weight_acc = tf.reduce_mean(tf.cast(tf.equal(weight_pred, tf.cast(batch["weight_label"], tf.int32)), tf.float32))
        rescue_pred = tf.cast(tf.sigmoid(out["rescue_logits"]) > 0.5, tf.float32)
        rescue_acc = tf.reduce_mean(tf.cast(tf.equal(rescue_pred, tf.reshape(tf.cast(batch["rescueable"], tf.float32), [-1])), tf.float32))
        std_p = tf.sigmoid(out["standard_logits"])
        brier = tf.reduce_mean(tf.square(std_p - tf.reshape(tf.cast(batch["standard_reachable"], tf.float32), [-1])))
        return loss, weight_acc, rescue_acc, brier, cand_top1, cand_pos

    hist_dir = out_dir / "training"
    hist_dir.mkdir(parents=True, exist_ok=True)
    hist_path = hist_dir / "training_history.csv"
    fieldnames = ["epoch", "train_loss", "val_loss", "val_weight_acc", "val_rescue_acc", "val_std_brier",
                  "val_candidate_top1_acc", "val_candidate_positive_rate", "lr", "train_samples_seen", "val_samples_seen", "framework", "num_gpus"]
    best_val = float("inf")
    epochs = int(tcfg.get("epochs", 48))
    with hist_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            # cosine decay manually
            lr_now = lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * (epoch - 1) / max(1, epochs))))
            opt.learning_rate.assign(float(lr_now))
            train_losses = []
            for batch in train_ds:
                loss, _, _ = train_step(batch)
                train_losses.append(float(loss.numpy()))
            vals = {"loss": [], "weight": [], "rescue": [], "brier": [], "cand": [], "pos": []}
            for batch in val_ds:
                loss, wacc, racc, brier, cacc, cpos = val_step(batch)
                vals["loss"].append(float(loss.numpy()))
                vals["weight"].append(float(wacc.numpy()))
                vals["rescue"].append(float(racc.numpy()))
                vals["brier"].append(float(brier.numpy()))
                vals["cand"].append(float(cacc.numpy()))
                vals["pos"].append(float(cpos.numpy()))
            rec = {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": float(np.mean(vals["loss"])),
                "val_weight_acc": float(np.mean(vals["weight"])),
                "val_rescue_acc": float(np.mean(vals["rescue"])),
                "val_std_brier": float(np.mean(vals["brier"])),
                "val_candidate_top1_acc": float(np.mean(vals["cand"])),
                "val_candidate_positive_rate": float(np.mean(vals["pos"])),
                "lr": float(lr_now),
                "train_samples_seen": int(train_arr["weight_label"].shape[0]),
                "val_samples_seen": int(val_arr["weight_label"].shape[0]),
                "framework": "tensorflow",
                "num_gpus": len(tf.config.list_physical_devices("GPU")),
            }
            writer.writerow(rec); f.flush()
            print(f"Epoch {epoch} | train_loss={rec['train_loss']:.4f} val_loss={rec['val_loss']:.4f} cand_top1={rec['val_candidate_top1_acc']:.4f} rescue_acc={rec['val_rescue_acc']:.4f} gpu={rec['num_gpus']}", flush=True)
            model.save_weights(str(weights_path))
            if rec["val_loss"] < best_val:
                best_val = rec["val_loss"]
                model.save_weights(str(best_path))
                print(f"New best checkpoint at epoch {epoch} val_loss={best_val:.4f}", flush=True)
    summary = {
        "epochs": epochs,
        "framework": "tensorflow",
        "checkpoint": str(weights_path),
        "best_checkpoint": str(best_path),
        "best_val_loss": best_val,
        "num_gpus": len(tf.config.list_physical_devices("GPU")),
    }
    (hist_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (ckpt_dir / "rescue_net_tf_meta.json").write_text(json.dumps({"code": code.code_summary(), "config": cfg}, indent=2, default=str), encoding="utf-8")
