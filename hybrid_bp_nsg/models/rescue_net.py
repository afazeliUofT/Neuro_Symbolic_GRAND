from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

try:  # TensorFlow is intentionally imported lazily/optionally for HPC-safe installs.
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None  # type: ignore


def require_tf():
    if tf is None:  # pragma: no cover
        raise RuntimeError(
            "TensorFlow is required for v12 TensorFlow/Keras rescue training/inference. "
            "Activate the FIR .venv that contains tensorflow before running train/evaluate."
        )
    return tf


def _compute_dtype(layer) -> "tf.dtypes.DType":
    require_tf()
    try:
        return tf.as_dtype(getattr(layer, "compute_dtype", None) or tf.keras.backend.floatx())
    except Exception:
        return tf.float32


class CandidateReranker(tf.keras.Model if tf is not None else object):  # type: ignore[misc]
    def __init__(self, packet_dim: int, candidate_feature_dim: int = 8, hidden_dim: int = 128, name: str = "candidate_reranker"):
        require_tf()
        super().__init__(name=name)
        self.packet_dim = int(packet_dim)
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.d1 = tf.keras.layers.Dense(hidden_dim, activation="gelu")
        self.n1 = tf.keras.layers.LayerNormalization()
        self.d2 = tf.keras.layers.Dense(max(8, hidden_dim // 2), activation="gelu")
        # Force final scores to float32 even when mixed precision is enabled.
        self.out = tf.keras.layers.Dense(1, dtype="float32")

    def call(self, packet_embedding, candidate_features, training: bool = False):
        comp_dtype = _compute_dtype(self)
        c = tf.shape(candidate_features)[1]
        pkt = tf.cast(tf.tile(packet_embedding[:, None, :], [1, c, 1]), comp_dtype)
        cand = tf.cast(candidate_features, comp_dtype)
        x = tf.concat([pkt, cand], axis=-1)
        x = self.d1(x)
        x = self.n1(x)
        x = self.d2(x)
        return tf.cast(tf.squeeze(self.out(x), axis=-1), tf.float32)


class TransformerBlock(tf.keras.layers.Layer if tf is not None else object):  # type: ignore[misc]
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.05, name: str | None = None):
        require_tf()
        super().__init__(name=name)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = max(1, int(num_heads))
        key_dim = max(8, int(hidden_dim) // self.num_heads)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=key_dim, dropout=float(dropout))
        self.drop1 = tf.keras.layers.Dropout(float(dropout))
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.ff1 = tf.keras.layers.Dense(4 * int(hidden_dim), activation="gelu")
        self.ff2 = tf.keras.layers.Dense(int(hidden_dim))
        self.drop2 = tf.keras.layers.Dropout(float(dropout))

    def call(self, x, training: bool = False):
        y = self.norm1(x)
        y = self.attn(y, y, training=training)
        x = x + self.drop1(y, training=training)
        y = self.norm2(x)
        y = self.ff2(self.ff1(y))
        x = x + self.drop2(y, training=training)
        return x


class RescueNet(tf.keras.Model if tf is not None else object):  # type: ignore[misc]
    """TensorFlow/Keras code-aware graph network for channel-aligned GRAND rescue.

    The implementation keeps mixed precision optional for FIR H100 runs, but all
    custom tf ops inside ``call`` explicitly harmonize dtypes so graph mode / XLA
    can run without float16/float32 mismatches.
    """

    def __init__(
        self,
        num_var_features: int,
        num_check_features: int,
        num_global_features: int,
        n: int,
        m: int,
        num_segments: int = 8,
        max_weight_class: int = 48,
        hidden_dim: int = 128,
        graph_layers: int = 5,
        top_k_tokens: int = 64,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        dropout: float = 0.05,
        candidate_feature_dim: int = 8,
        h_dense: Optional[np.ndarray] = None,
        deg_v: Optional[np.ndarray] = None,
        deg_c: Optional[np.ndarray] = None,
        name: str = "tf_rescue_net",
    ):
        require_tf()
        super().__init__(name=name)
        self.n = int(n)
        self.m = int(m)
        self.num_segments = int(num_segments)
        self.max_weight_class = int(max_weight_class)
        self.hidden_dim = int(hidden_dim)
        self.graph_layers = int(graph_layers)
        self.top_k_tokens = int(top_k_tokens)
        self.candidate_feature_dim = int(candidate_feature_dim)

        self.var_dense = tf.keras.layers.Dense(hidden_dim, activation="gelu")
        self.var_norm = tf.keras.layers.LayerNormalization()
        self.var_drop = tf.keras.layers.Dropout(float(dropout))

        self.check_dense = tf.keras.layers.Dense(hidden_dim, activation="gelu")
        self.check_norm = tf.keras.layers.LayerNormalization()
        self.check_drop = tf.keras.layers.Dropout(float(dropout))

        self.global_dense = tf.keras.layers.Dense(hidden_dim, activation="gelu")
        self.global_norm = tf.keras.layers.LayerNormalization()

        self.v_updates = []
        self.c_updates = []
        self.v_norms = []
        self.c_norms = []
        self.dropouts = []
        for i in range(self.graph_layers):
            self.v_updates.append(tf.keras.Sequential([
                tf.keras.layers.Dense(hidden_dim, activation="gelu"),
                tf.keras.layers.Dense(hidden_dim),
            ], name=f"v_update_{i}"))
            self.c_updates.append(tf.keras.Sequential([
                tf.keras.layers.Dense(hidden_dim, activation="gelu"),
                tf.keras.layers.Dense(hidden_dim),
            ], name=f"c_update_{i}"))
            self.v_norms.append(tf.keras.layers.LayerNormalization(name=f"v_norm_{i}"))
            self.c_norms.append(tf.keras.layers.LayerNormalization(name=f"c_norm_{i}"))
            self.dropouts.append(tf.keras.layers.Dropout(float(dropout)))

        self.token_blocks = [
            TransformerBlock(hidden_dim, transformer_heads, dropout, name=f"token_block_{i}")
            for i in range(max(1, int(transformer_layers)))
        ]

        self.bit_d1 = tf.keras.layers.Dense(max(8, hidden_dim // 2), activation="gelu")
        self.bit_out = tf.keras.layers.Dense(1, dtype="float32")

        self.packet_d1 = tf.keras.layers.Dense(hidden_dim, activation="gelu")
        self.packet_norm = tf.keras.layers.LayerNormalization()

        self.segment_head = tf.keras.layers.Dense(num_segments, dtype="float32")
        self.weight_head = tf.keras.layers.Dense(max_weight_class + 2, dtype="float32")
        self.standard_head = tf.keras.layers.Dense(1, dtype="float32")
        self.expanded_head = tf.keras.layers.Dense(1, dtype="float32")
        self.rescue_head = tf.keras.layers.Dense(1, dtype="float32")
        self.reranker = CandidateReranker(hidden_dim, candidate_feature_dim=candidate_feature_dim, hidden_dim=hidden_dim)

        self.set_graph(h_dense, deg_v, deg_c)

    def set_graph(self, h_dense: Optional[np.ndarray], deg_v: Optional[np.ndarray], deg_c: Optional[np.ndarray]) -> None:
        if h_dense is None:
            h_dense = np.zeros((self.m, self.n), dtype=np.float32)
        if deg_v is None:
            deg_v = np.ones((self.n,), dtype=np.float32)
        if deg_c is None:
            deg_c = np.ones((self.m,), dtype=np.float32)
        self.h_dense = tf.constant(np.asarray(h_dense, dtype=np.float32))
        self.deg_v = tf.constant(np.maximum(np.asarray(deg_v, dtype=np.float32), 1.0))
        self.deg_c = tf.constant(np.maximum(np.asarray(deg_c, dtype=np.float32), 1.0))

    def call(self, inputs: Dict[str, Any], training: bool = False) -> Dict[str, Any]:
        var_features = tf.cast(inputs["var_features"], tf.float32)
        check_features = tf.cast(inputs["check_features"], tf.float32)
        global_features = tf.cast(inputs["global_features"], tf.float32)
        heuristic_order = tf.cast(inputs["heuristic_order"], tf.int32)

        v = self.var_drop(self.var_norm(self.var_dense(var_features)), training=training)
        c = self.check_drop(self.check_norm(self.check_dense(check_features)), training=training)

        msg_dtype = v.dtype
        h = tf.cast(self.h_dense, msg_dtype)
        dv = tf.cast(self.deg_v, msg_dtype)
        dc = tf.cast(self.deg_c, msg_dtype)

        for vu, cu, vn, cn, drop in zip(self.v_updates, self.c_updates, self.v_norms, self.c_norms, self.dropouts):
            c_to_v = tf.einsum("mn,bmh->bnh", h, tf.cast(c, msg_dtype)) / dv[None, :, None]
            v_to_c = tf.einsum("mn,bnh->bmh", h, tf.cast(v, msg_dtype)) / dc[None, :, None]
            v_in = tf.concat([tf.cast(v, msg_dtype), c_to_v], axis=-1)
            c_in = tf.concat([tf.cast(c, msg_dtype), v_to_c], axis=-1)
            v = vn(tf.cast(v, msg_dtype) + drop(vu(v_in, training=training), training=training))
            c = cn(tf.cast(c, msg_dtype) + drop(cu(c_in, training=training), training=training))

        bit_logits = tf.cast(tf.squeeze(self.bit_out(self.bit_d1(v)), axis=-1), tf.float32)

        bsz = tf.shape(v)[0]
        n = tf.shape(v)[1]
        k = tf.minimum(tf.cast(self.top_k_tokens, tf.int32), n)
        idx = tf.clip_by_value(heuristic_order[:, :k], 0, n - 1)
        tokens = tf.gather(v, idx, batch_dims=1)
        for block in self.token_blocks:
            tokens = block(tokens, training=training)

        mean_pool = tf.reduce_mean(v, axis=1)
        max_pool = tf.reduce_max(v, axis=1)
        token_pool = tf.reduce_mean(tokens, axis=1)
        g = self.global_norm(self.global_dense(global_features))
        packet = self.packet_norm(self.packet_d1(tf.concat([mean_pool + tf.cast(g, mean_pool.dtype), max_pool, token_pool], axis=-1)))
        packet_f32 = tf.cast(packet, tf.float32)

        out = {
            "bit_logits": bit_logits,
            "segment_logits": tf.cast(self.segment_head(packet), tf.float32),
            "weight_logits": tf.cast(self.weight_head(packet), tf.float32),
            "standard_logits": tf.cast(tf.squeeze(self.standard_head(packet), axis=-1), tf.float32),
            "expanded_logits": tf.cast(tf.squeeze(self.expanded_head(packet), axis=-1), tf.float32),
            "rescue_logits": tf.cast(tf.squeeze(self.rescue_head(packet), axis=-1), tf.float32),
            "packet_embedding": packet_f32,
        }
        if "candidate_features" in inputs:
            out["candidate_scores"] = self.reranker(packet_f32, inputs["candidate_features"], training=training)
        else:
            out["candidate_scores"] = tf.zeros((bsz, 0), dtype=tf.float32)
        return out


def build_rescue_net_from_shapes(shapes: Dict[str, int], cfg: Dict[str, Any], code) -> RescueNet:
    """Build and graph-initialize the Keras RescueNet from dataset shapes and code."""
    return RescueNet(
        num_var_features=int(shapes["num_var_features"]),
        num_check_features=int(shapes["num_check_features"]),
        num_global_features=int(shapes["num_global_features"]),
        n=int(code.n),
        m=int(code.m),
        num_segments=int(cfg["model"]["num_segments"]),
        max_weight_class=int(cfg["model"]["max_weight_class"]),
        hidden_dim=int(cfg["model"]["graph_hidden_dim"]),
        graph_layers=int(cfg["model"]["graph_layers"]),
        top_k_tokens=int(cfg["model"]["top_k_tokens"]),
        transformer_heads=int(cfg["model"].get("transformer_heads", 4)),
        transformer_layers=int(cfg["model"].get("transformer_layers", 1)),
        dropout=float(cfg["train"].get("dropout", 0.05)),
        candidate_feature_dim=int(shapes.get("candidate_feature_dim", 8)),
        h_dense=code.h.astype(np.float32),
        deg_v=np.maximum(code.deg_v.astype(np.float32), 1.0),
        deg_c=np.maximum(code.deg_c.astype(np.float32), 1.0),
    )
