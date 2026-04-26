from __future__ import annotations

from typing import Dict

import numpy as np


def build_rescue_net(code, cfg: Dict, var_dim: int = 16, check_dim: int = 6, global_dim: int = 9, candidate_dim: int = 14):
    import tensorflow as tf

    class CandidateReranker(tf.keras.layers.Layer):
        def __init__(self, hidden_dim: int, dropout: float):
            super().__init__()
            self.d1 = tf.keras.layers.Dense(hidden_dim, activation="gelu")
            self.drop = tf.keras.layers.Dropout(dropout)
            self.d2 = tf.keras.layers.Dense(hidden_dim // 2, activation="gelu")
            self.out = tf.keras.layers.Dense(1, dtype="float32")

        def call(self, packet_embedding, candidate_features, training: bool = False):
            c = tf.shape(candidate_features)[1]
            pkt = tf.tile(packet_embedding[:, None, :], [1, c, 1])
            x = tf.concat([tf.cast(pkt, tf.float32), tf.cast(candidate_features, tf.float32)], axis=-1)
            x = self.d1(x)
            x = self.drop(x, training=training)
            x = self.d2(x)
            return tf.squeeze(self.out(x), axis=-1)

    class RescueNet(tf.keras.Model):
        def __init__(self):
            super().__init__(name="tf_rescue_net_v14")
            mcfg = cfg.get("model", {})
            tcfg = cfg.get("train", {})
            hidden_dim = int(mcfg.get("graph_hidden_dim", 128))
            self.graph_layers = int(mcfg.get("graph_layers", 5))
            self.top_k_tokens = int(mcfg.get("top_k_tokens", 64))
            self.h_mat = tf.constant(code.h.astype(np.float32), dtype=tf.float32)
            self.deg_v = tf.constant(np.maximum(code.deg_v.astype(np.float32), 1.0), dtype=tf.float32)
            self.deg_c = tf.constant(np.maximum(code.deg_c.astype(np.float32), 1.0), dtype=tf.float32)
            drop_rate = float(tcfg.get("dropout", 0.08))
            self.v_in = tf.keras.layers.Dense(hidden_dim, activation="gelu")
            self.c_in = tf.keras.layers.Dense(hidden_dim, activation="gelu")
            self.g_in = tf.keras.layers.Dense(hidden_dim, activation="gelu")
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
                self.dropouts.append(tf.keras.layers.Dropout(drop_rate))
            self.packet_d1 = tf.keras.layers.Dense(hidden_dim, activation="gelu")
            self.packet_norm = tf.keras.layers.LayerNormalization()
            self.bit_d1 = tf.keras.layers.Dense(hidden_dim, activation="gelu")
            self.bit_out = tf.keras.layers.Dense(1, dtype="float32")
            self.segment_head = tf.keras.Sequential([
                tf.keras.layers.Dense(hidden_dim, activation="gelu"),
                tf.keras.layers.Dense(int(mcfg.get("num_segments", 8)), dtype="float32"),
            ])
            self.weight_head = tf.keras.Sequential([
                tf.keras.layers.Dense(hidden_dim, activation="gelu"),
                tf.keras.layers.Dense(int(mcfg.get("max_weight_class", 64)) + 1, dtype="float32"),
            ])
            self.standard_head = tf.keras.layers.Dense(1, dtype="float32")
            self.expanded_head = tf.keras.layers.Dense(1, dtype="float32")
            self.rescue_head = tf.keras.layers.Dense(1, dtype="float32")
            self.reranker = CandidateReranker(hidden_dim, drop_rate)

        def call(self, inputs: Dict[str, object], training: bool = False):
            var_features = tf.cast(inputs["var_features"], tf.float32)
            check_features = tf.cast(inputs["check_features"], tf.float32)
            global_features = tf.cast(inputs["global_features"], tf.float32)
            v = self.v_in(var_features)
            c = self.c_in(check_features)
            h = self.h_mat
            dv = self.deg_v
            dc = self.deg_c
            for vu, cu, vn, cn, drop in zip(self.v_updates, self.c_updates, self.v_norms, self.c_norms, self.dropouts):
                c_to_v = tf.einsum("mn,bmh->bnh", h, c) / dv[None, :, None]
                v_to_c = tf.einsum("mn,bnh->bmh", h, v) / dc[None, :, None]
                v = vn(v + drop(vu(tf.concat([v, c_to_v], axis=-1), training=training), training=training))
                c = cn(c + drop(cu(tf.concat([c, v_to_c], axis=-1), training=training), training=training))
            bit_logits = tf.squeeze(self.bit_out(self.bit_d1(v)), axis=-1)
            mean_pool = tf.reduce_mean(v, axis=1)
            max_pool = tf.reduce_max(v, axis=1)
            # top-k variable tokens by learned/suspicion-rich feature channel if available.
            score = var_features[:, :, -1]
            k = tf.minimum(tf.shape(v)[1], self.top_k_tokens)
            idx = tf.nn.top_k(score, k=k).indices
            tokens = tf.gather(v, idx, batch_dims=1)
            token_pool = tf.reduce_mean(tokens, axis=1)
            check_mean_pool = tf.reduce_mean(c, axis=1)
            check_max_pool = tf.reduce_max(c, axis=1)
            g = self.g_in(global_features)
            packet = self.packet_norm(self.packet_d1(tf.concat([
                mean_pool + tf.cast(g, mean_pool.dtype),
                max_pool,
                token_pool,
                check_mean_pool,
                check_max_pool,
            ], axis=-1)))
            packet_f32 = tf.cast(packet, tf.float32)
            out = {
                "bit_logits": tf.cast(bit_logits, tf.float32),
                "segment_logits": tf.cast(self.segment_head(packet), tf.float32),
                "weight_logits": tf.cast(self.weight_head(packet), tf.float32),
                "standard_logits": tf.cast(tf.squeeze(self.standard_head(packet), axis=-1), tf.float32),
                "expanded_logits": tf.cast(tf.squeeze(self.expanded_head(packet), axis=-1), tf.float32),
                "rescue_logits": tf.cast(tf.squeeze(self.rescue_head(packet), axis=-1), tf.float32),
                "packet_embedding": packet_f32,
            }
            if "candidate_features" in inputs:
                out["candidate_scores"] = self.reranker(packet_f32, inputs["candidate_features"], training=training)
            return out

    return RescueNet()
