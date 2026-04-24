from __future__ import annotations

import csv
import os
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from ..codes.factory import build_code
from ..models.rescue_net import build_rescue_net_from_shapes
from ..training.dataset import NPZShardDataset, infer_shapes
from ..utils.io import ensure_dir, write_json
from ..utils.logging import get_logger
from ..utils.tf_gpu import configure_tensorflow


def _to_tf_batch(tf, batch: Dict[str, np.ndarray]) -> Dict[str, object]:
    float_keys = [
        "var_features", "check_features", "global_features", "candidate_features",
        "bit_labels", "segment_labels", "standard_reachable", "expanded_reachable",
        "rescueable", "candidate_labels", "candidate_valid",
    ]
    int_keys = ["heuristic_order", "weight_label"]
    out = {}
    for k in float_keys:
        if k in batch:
            out[k] = tf.convert_to_tensor(batch[k], dtype=tf.float32)
    for k in int_keys:
        if k in batch:
            out[k] = tf.convert_to_tensor(batch[k], dtype=tf.int32)
    return out


def _weighted_bce(tf, logits, labels, pos_weight: float):
    logits = tf.cast(logits, tf.float32)
    labels = tf.cast(labels, tf.float32)
    return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=labels, logits=logits, pos_weight=float(pos_weight)))


def _bit_rank_loss(tf, bit_logits, bit_labels):
    bit_logits = tf.cast(bit_logits, tf.float32)
    bit_labels = tf.cast(bit_labels, tf.float32)
    pos = bit_labels > 0.5
    neg = tf.logical_not(pos)
    pos_count = tf.reduce_sum(tf.cast(pos, tf.float32), axis=1)
    neg_count = tf.reduce_sum(tf.cast(neg, tf.float32), axis=1)
    pos_sum = tf.reduce_sum(tf.where(pos, bit_logits, tf.zeros_like(bit_logits)), axis=1)
    pos_mean = pos_sum / tf.maximum(pos_count, 1.0)
    neg_logits = tf.where(neg, bit_logits, tf.fill(tf.shape(bit_logits), tf.constant(-1e9, dtype=tf.float32)))
    k = min(16, int(bit_logits.shape[-1]) if bit_logits.shape[-1] is not None else 16)
    neg_top = tf.nn.top_k(neg_logits, k=k).values
    neg_mean = tf.reduce_mean(neg_top, axis=1)
    valid = tf.logical_and(pos_count > 0, neg_count > 0)
    rank = tf.nn.relu(1.0 - pos_mean + neg_mean)
    return tf.reduce_sum(tf.where(valid, rank, tf.zeros_like(rank))) / tf.maximum(tf.reduce_sum(tf.cast(valid, tf.float32)), 1.0)


def _candidate_stats(tf, cand_scores, cand_labels, cand_valid):
    cand_valid = tf.cast(cand_valid, tf.float32)
    cand_labels = tf.cast(cand_labels, tf.float32)
    positive_exists = tf.reduce_sum(cand_labels * cand_valid, axis=1) > 0.0
    masked_scores = tf.where(cand_valid > 0.0, tf.cast(cand_scores, tf.float32), tf.fill(tf.shape(cand_scores), tf.constant(-1e9, dtype=tf.float32)))
    target_idx = tf.argmax(cand_labels, axis=1, output_type=tf.int32)

    def positive_loss():
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            tf.boolean_mask(target_idx, positive_exists),
            tf.boolean_mask(masked_scores, positive_exists),
            from_logits=True,
        ))

    def positive_acc():
        top1 = tf.argmax(masked_scores, axis=1, output_type=tf.int32)
        return tf.reduce_mean(tf.cast(tf.equal(tf.boolean_mask(top1, positive_exists), tf.boolean_mask(target_idx, positive_exists)), tf.float32))

    loss = tf.cond(tf.reduce_any(positive_exists), positive_loss, lambda: tf.constant(0.0, dtype=tf.float32))
    acc = tf.cond(tf.reduce_any(positive_exists), positive_acc, lambda: tf.constant(0.0, dtype=tf.float32))
    pos_rate = tf.reduce_mean(tf.cast(positive_exists, tf.float32))
    return loss, acc, pos_rate


def _multitask_loss(tf, cfg: Dict[str, object], model, out: Dict[str, object], batch: Dict[str, object]) -> Tuple[object, Dict[str, object]]:
    lw = {k: float(v) for k, v in cfg["train"]["loss_weights"].items()}
    bit_pos_w = float(cfg["train"].get("bit_pos_weight", 4.0))
    reach_pos_w = float(cfg["train"].get("reach_pos_weight", 4.0))

    losses = {}
    losses["bit"] = _weighted_bce(tf, out["bit_logits"], batch["bit_labels"], bit_pos_w)
    losses["segment"] = _weighted_bce(tf, out["segment_logits"], batch["segment_labels"], reach_pos_w)
    losses["weight"] = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
        tf.reshape(tf.cast(batch["weight_label"], tf.int32), [-1]),
        tf.cast(out["weight_logits"], tf.float32),
        from_logits=True,
    ))
    losses["rank"] = _bit_rank_loss(tf, out["bit_logits"], batch["bit_labels"])
    losses["standard_reachable"] = _weighted_bce(tf, out["standard_logits"], tf.reshape(batch["standard_reachable"], [-1]), reach_pos_w)
    losses["expanded_reachable"] = _weighted_bce(tf, out["expanded_logits"], tf.reshape(batch["expanded_reachable"], [-1]), reach_pos_w)
    losses["rescueable"] = _weighted_bce(tf, out["rescue_logits"], tf.reshape(batch["rescueable"], [-1]), reach_pos_w)

    cand_scores = model.reranker(out["packet_embedding"], batch["candidate_features"], training=True)
    losses["rerank"], cand_top1_acc, cand_pos_rate = _candidate_stats(tf, cand_scores, batch["candidate_labels"], batch["candidate_valid"])

    total = tf.constant(0.0, dtype=tf.float32)
    for k, v in losses.items():
        total = total + float(lw.get(k, 0.0)) * tf.cast(v, tf.float32)
    aux = {
        "candidate_top1_acc": cand_top1_acc,
        "candidate_positive_rate": cand_pos_rate,
    }
    return total, losses | aux


def train_model(cfg: Dict[str, object]) -> None:
    logger = get_logger("train")
    out_dir = ensure_dir(Path(cfg["project"]["output_dir"]))
    train_dir = out_dir / "datasets" / "train"
    val_dir = out_dir / "datasets" / "val"
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    training_dir = ensure_dir(out_dir / "training")
    hist_path = training_dir / "training_history.csv"
    write_json(cfg, out_dir / "artifacts" / "resolved_config.json")

    tf, gpus = configure_tensorflow(
        require_gpu=bool(cfg["train"].get("require_gpu", False)),
        mixed_precision=bool(cfg["train"].get("mixed_precision", True)),
        xla=bool(cfg["train"].get("xla", True)),
        cpu_threads=int(cfg["train"].get("cpu_threads", os.environ.get("SLURM_CPUS_PER_TASK", 0) or 0)),
    )
    logger.info("TensorFlow version=%s GPUs=%s mixed_policy=%s", tf.__version__, gpus, tf.keras.mixed_precision.global_policy())

    shapes = infer_shapes(train_dir)
    code = build_code(cfg["code"])
    model = build_rescue_net_from_shapes(shapes, cfg, code)

    train_ds = NPZShardDataset(train_dir, preload=bool(cfg["train"].get("preload_dataset", True)))
    val_ds = NPZShardDataset(val_dir, preload=bool(cfg["train"].get("preload_dataset", True)))
    batch_size = int(cfg["train"]["batch_size"])
    rng = np.random.default_rng(int(cfg["project"].get("seed", 1234)) + 4242)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=float(cfg["train"]["lr"]),
        decay_steps=max(1, int(cfg["train"]["epochs"]) * max(1, len(train_ds) // max(1, batch_size))),
        alpha=float(cfg["train"].get("lr_alpha", 0.05)),
    )
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=float(cfg["train"].get("weight_decay", 1e-5)))

    first_batch_np = next(train_ds.iter_batches(batch_size=min(batch_size, len(train_ds)), shuffle=False))
    first_batch = _to_tf_batch(tf, first_batch_np)
    _ = model({
        "var_features": first_batch["var_features"],
        "check_features": first_batch["check_features"],
        "global_features": first_batch["global_features"],
        "heuristic_order": first_batch["heuristic_order"],
        "candidate_features": first_batch["candidate_features"],
    }, training=False)

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, str(ckpt_dir / "tf_ckpt"), max_to_keep=3)
    meta_path = ckpt_dir / "rescue_net_tf_meta.json"
    weights_path = ckpt_dir / "rescue_net_tf.weights.h5"
    best_weights_path = ckpt_dir / "rescue_net_tf.best.weights.h5"
    start_epoch = 1
    best_val = float("inf")
    best_epoch = 0
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            best_val = float(meta.get("best_val_loss", best_val))
            best_epoch = int(meta.get("best_epoch", best_epoch))
        except Exception:
            pass
    if bool(cfg["train"].get("resume", True)) and manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        if meta_path.exists():
            try:
                start_epoch = int(json.loads(meta_path.read_text()).get("epoch", 0)) + 1
            except Exception:
                start_epoch = 1
        logger.info("Resumed TensorFlow checkpoint %s at epoch %d", manager.latest_checkpoint, start_epoch)
    elif bool(cfg["train"].get("resume", True)) and weights_path.exists():
        model.load_weights(str(weights_path))
        if meta_path.exists():
            try:
                start_epoch = int(json.loads(meta_path.read_text()).get("epoch", 0)) + 1
            except Exception:
                start_epoch = 1
        logger.info("Resumed Keras weights %s at epoch %d", weights_path, start_epoch)

    grad_clip = float(cfg["train"].get("grad_clip", 0.0))
    jit_compile = bool(cfg["train"].get("xla", True) and len(gpus) > 0)

    @tf.function(reduce_retracing=True, jit_compile=jit_compile)
    def train_step(batch):
        with tf.GradientTape() as tape:
            out = model({
                "var_features": batch["var_features"],
                "check_features": batch["check_features"],
                "global_features": batch["global_features"],
                "heuristic_order": batch["heuristic_order"],
                "candidate_features": batch["candidate_features"],
            }, training=True)
            loss, aux = _multitask_loss(tf, cfg, model, out, batch)
        grads = tape.gradient(loss, model.trainable_variables)
        if grad_clip > 0:
            grads, _ = tf.clip_by_global_norm(grads, grad_clip)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, aux

    @tf.function(reduce_retracing=True, jit_compile=jit_compile)
    def val_step(batch):
        out = model({
            "var_features": batch["var_features"],
            "check_features": batch["check_features"],
            "global_features": batch["global_features"],
            "heuristic_order": batch["heuristic_order"],
            "candidate_features": batch["candidate_features"],
        }, training=False)
        loss, aux = _multitask_loss(tf, cfg, model, out, batch)
        pred_w = tf.argmax(out["weight_logits"], axis=-1, output_type=tf.int32)
        weight_acc = tf.reduce_mean(tf.cast(tf.equal(pred_w, tf.reshape(tf.cast(batch["weight_label"], tf.int32), [-1])), tf.float32))
        rescue_pred = tf.cast(tf.sigmoid(tf.cast(out["rescue_logits"], tf.float32)) > 0.5, tf.float32)
        rescue_acc = tf.reduce_mean(tf.cast(tf.equal(rescue_pred, tf.reshape(tf.cast(batch["rescueable"], tf.float32), [-1])), tf.float32))
        std_p = tf.sigmoid(tf.cast(out["standard_logits"], tf.float32))
        brier = tf.reduce_mean(tf.square(std_p - tf.reshape(tf.cast(batch["standard_reachable"], tf.float32), [-1])))
        return loss, weight_acc, rescue_acc, brier, aux["candidate_top1_acc"], aux["candidate_positive_rate"]

    exists = hist_path.exists()
    with hist_path.open("a", newline="", encoding="utf-8") as f:
        fieldnames = [
            "epoch", "train_loss", "val_loss", "val_weight_acc", "val_rescue_acc", "val_std_brier",
            "val_candidate_top1_acc", "val_candidate_positive_rate", "lr", "train_samples_seen", "val_samples_seen", "framework", "num_gpus"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()

        for epoch in range(start_epoch, int(cfg["train"]["epochs"]) + 1):
            train_losses = []
            for batch_np in train_ds.iter_batches(batch_size=batch_size, shuffle=True, rng=rng):
                batch = _to_tf_batch(tf, batch_np)
                loss, _ = train_step(batch)
                train_losses.append(float(loss.numpy()))

            val_losses = []
            weight_accs = []
            rescue_accs = []
            briers = []
            cand_accs = []
            cand_pos_rates = []
            for batch_np in val_ds.iter_batches(batch_size=batch_size, shuffle=False):
                batch = _to_tf_batch(tf, batch_np)
                loss, wa, ra, br, ca, cpr = val_step(batch)
                val_losses.append(float(loss.numpy()))
                weight_accs.append(float(wa.numpy()))
                rescue_accs.append(float(ra.numpy()))
                briers.append(float(br.numpy()))
                cand_accs.append(float(ca.numpy()))
                cand_pos_rates.append(float(cpr.numpy()))

            lr_val = optimizer.learning_rate
            try:
                lr_now = float(lr_val.numpy())
            except Exception:
                lr_now = float(cfg["train"]["lr"])

            record = {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                "val_loss": float(np.mean(val_losses)) if val_losses else float("nan"),
                "val_weight_acc": float(np.mean(weight_accs)) if weight_accs else float("nan"),
                "val_rescue_acc": float(np.mean(rescue_accs)) if rescue_accs else float("nan"),
                "val_std_brier": float(np.mean(briers)) if briers else float("nan"),
                "val_candidate_top1_acc": float(np.mean(cand_accs)) if cand_accs else float("nan"),
                "val_candidate_positive_rate": float(np.mean(cand_pos_rates)) if cand_pos_rates else float("nan"),
                "lr": lr_now,
                "train_samples_seen": len(train_ds),
                "val_samples_seen": len(val_ds),
                "framework": "tensorflow",
                "num_gpus": len(gpus),
            }
            writer.writerow(record)
            f.flush()
            logger.info(
                "Epoch %d | train_loss=%.4f val_loss=%.4f cand_top1=%.4f rescue_acc=%.4f gpu=%d",
                epoch, record["train_loss"], record["val_loss"], record["val_candidate_top1_acc"], record["val_rescue_acc"], len(gpus)
            )

            manager.save(checkpoint_number=epoch)
            model.save_weights(str(weights_path))
            if record["val_loss"] < best_val:
                best_val = float(record["val_loss"])
                best_epoch = int(epoch)
                model.save_weights(str(best_weights_path))
                logger.info("New best checkpoint at epoch %d val_loss=%.4f", best_epoch, best_val)
            write_json({
                "epoch": epoch,
                "cfg": cfg,
                "shapes": shapes,
                "framework": "tensorflow",
                "weights": str(weights_path),
                "best_weights": str(best_weights_path),
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
            }, meta_path)

    write_json({
        "epochs": int(cfg["train"]["epochs"]),
        "framework": "tensorflow",
        "checkpoint": str(weights_path),
        "best_checkpoint": str(best_weights_path),
        "tf_checkpoint": manager.latest_checkpoint,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "num_gpus": len(gpus),
    }, training_dir / "training_summary.json")
