from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import torch
from torch import nn


class BipartiteLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.v2c = nn.Linear(hidden_dim, hidden_dim)
        self.c2v = nn.Linear(hidden_dim, hidden_dim)
        self.c_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.v_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.v_norm = nn.LayerNorm(hidden_dim)
        self.c_norm = nn.LayerNorm(hidden_dim)

    def forward(self, v: torch.Tensor, c: torch.Tensor, h_dense: torch.Tensor, deg_v: torch.Tensor, deg_c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c_msg = torch.einsum("mn,bnh->bmh", h_dense, self.v2c(v)) / deg_c[None, :, None]
        c = self.c_norm(c + self.c_mlp(torch.cat([c, c_msg], dim=-1)))
        v_msg = torch.einsum("nm,bmh->bnh", h_dense.t(), self.c2v(c)) / deg_v[None, :, None]
        v = self.v_norm(v + self.v_mlp(torch.cat([v, v_msg], dim=-1)))
        return v, c


class CandidateReranker(nn.Module):
    def __init__(self, packet_dim: int, candidate_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(packet_dim + candidate_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, packet_emb: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        B, C, D = candidate_features.shape
        packet_expand = packet_emb[:, None, :].expand(B, C, packet_emb.shape[-1])
        x = torch.cat([packet_expand, candidate_features], dim=-1)
        return self.net(x).squeeze(-1)


class RescueNet(nn.Module):
    def __init__(self, num_var_features: int, num_check_features: int, num_global_features: int,
                 n: int, m: int, num_segments: int = 8, max_weight_class: int = 8,
                 hidden_dim: int = 64, graph_layers: int = 3, top_k_tokens: int = 24,
                 transformer_heads: int = 4, transformer_layers: int = 2, dropout: float = 0.1,
                 candidate_feature_dim: int = 8):
        super().__init__()
        self.n = int(n)
        self.m = int(m)
        self.num_segments = int(num_segments)
        self.max_weight_class = int(max_weight_class)
        self.top_k_tokens = int(top_k_tokens)
        self.var_in = nn.Linear(num_var_features, hidden_dim)
        self.check_in = nn.Linear(num_check_features, hidden_dim)
        self.global_in = nn.Linear(num_global_features, hidden_dim)
        self.layers = nn.ModuleList([BipartiteLayer(hidden_dim, dropout) for _ in range(graph_layers)])
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
        self.rank_embed = nn.Embedding(n, hidden_dim)
        self.bit_head = nn.Linear(hidden_dim, 1)
        self.segment_pool = nn.Linear(hidden_dim, hidden_dim)
        self.packet_proj = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.weight_head = nn.Linear(hidden_dim, max_weight_class + 2)
        self.standard_head = nn.Linear(hidden_dim, 1)
        self.expanded_head = nn.Linear(hidden_dim, 1)
        self.rescue_head = nn.Linear(hidden_dim, 1)
        self.segment_head = nn.Linear(hidden_dim, num_segments)
        self.reranker = CandidateReranker(packet_dim=hidden_dim, candidate_dim=candidate_feature_dim,
                                          hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, var_features: torch.Tensor, check_features: torch.Tensor, global_features: torch.Tensor,
                heuristic_order: torch.Tensor, h_dense: torch.Tensor, deg_v: torch.Tensor, deg_c: torch.Tensor,
                candidate_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        v = self.var_in(var_features)
        c = self.check_in(check_features)
        g = self.global_in(global_features)
        for layer in self.layers:
            v, c = layer(v, c, h_dense, deg_v, deg_c)
        bit_logits = self.bit_head(v).squeeze(-1)
        B = v.shape[0]
        top_k = min(self.top_k_tokens, self.n)
        top_tokens = []
        pos_ids = torch.arange(top_k, device=v.device)
        for b in range(B):
            order_b = heuristic_order[b, :top_k].long()
            tok = v[b, order_b] + self.rank_embed(pos_ids)
            top_tokens.append(tok)
        top_tokens = torch.stack(top_tokens, dim=0)
        trans = self.transformer(top_tokens)
        pooled_tokens = trans.mean(dim=1)
        pooled_v = v.mean(dim=1)
        pooled_c = c.mean(dim=1)
        packet_emb = self.packet_proj(torch.cat([pooled_v, pooled_c, pooled_tokens, g], dim=-1))
        seg_size = (self.n + self.num_segments - 1) // self.num_segments
        seg_embs = []
        for s in range(self.num_segments):
            start = s * seg_size
            stop = min((s + 1) * seg_size, self.n)
            seg_tok = top_tokens[:, start:stop] if start < top_k else pooled_tokens[:, None, :]
            seg_embs.append(seg_tok.mean(dim=1))
        seg_emb = torch.stack(seg_embs, dim=1).mean(dim=1)
        outputs = {
            "bit_logits": bit_logits,
            "weight_logits": self.weight_head(packet_emb),
            "standard_logits": self.standard_head(packet_emb).squeeze(-1),
            "expanded_logits": self.expanded_head(packet_emb).squeeze(-1),
            "rescue_logits": self.rescue_head(packet_emb).squeeze(-1),
            "segment_logits": self.segment_head(seg_emb),
            "packet_embedding": packet_emb,
        }
        if candidate_features is not None:
            outputs["candidate_scores"] = self.reranker(packet_emb, candidate_features)
        return outputs
