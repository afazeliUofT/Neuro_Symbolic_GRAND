from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


class CandidateReranker(nn.Module):
    def __init__(self, packet_dim: int, candidate_feature_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(packet_dim + candidate_feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, packet_embedding: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        # packet_embedding: [B,D], candidate_features: [B,C,F]
        b, c, _ = candidate_features.shape
        pkt = packet_embedding[:, None, :].expand(b, c, packet_embedding.shape[-1])
        x = torch.cat([pkt, candidate_features], dim=-1)
        return self.net(x).squeeze(-1)


class RescueNet(nn.Module):
    """Code-aware graph network for channel-aligned rescue ranking.

    The network is deliberately code-aware but modest: it exchanges messages through H,
    predicts per-bit correction probability on the GRAND base, predicts target weight,
    and scores candidate masks. It is not used as a skip gate by default.
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
    ):
        super().__init__()
        self.n = int(n)
        self.m = int(m)
        self.num_segments = int(num_segments)
        self.max_weight_class = int(max_weight_class)
        self.hidden_dim = int(hidden_dim)
        self.graph_layers = int(graph_layers)
        self.top_k_tokens = int(top_k_tokens)

        self.var_in = nn.Sequential(
            nn.Linear(num_var_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.check_in = nn.Sequential(
            nn.Linear(num_check_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.global_in = nn.Sequential(
            nn.Linear(num_global_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.v_updates = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(graph_layers)
        ])
        self.c_updates = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(graph_layers)
        ])
        self.v_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(graph_layers)])
        self.c_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(graph_layers)])
        self.dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(1, int(transformer_heads)),
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.token_encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, int(transformer_layers)))

        self.bit_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        packet_dim = hidden_dim * 3
        self.packet_proj = nn.Sequential(
            nn.Linear(packet_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.segment_head = nn.Linear(hidden_dim, num_segments)
        self.weight_head = nn.Linear(hidden_dim, max_weight_class + 2)
        self.standard_head = nn.Linear(hidden_dim, 1)
        self.expanded_head = nn.Linear(hidden_dim, 1)
        self.rescue_head = nn.Linear(hidden_dim, 1)
        self.reranker = CandidateReranker(hidden_dim, candidate_feature_dim=candidate_feature_dim, hidden_dim=hidden_dim)

    def forward(
        self,
        var_features: torch.Tensor,
        check_features: torch.Tensor,
        global_features: torch.Tensor,
        heuristic_order: torch.Tensor,
        h_dense: torch.Tensor,
        deg_v: torch.Tensor,
        deg_c: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Shapes: var [B,N,Fv], check [B,M,Fc], H [M,N]
        v = self.var_in(var_features)
        c = self.check_in(check_features)
        h = h_dense.to(v.device).float()
        dv = deg_v.to(v.device).float().clamp_min(1.0)
        dc = deg_c.to(v.device).float().clamp_min(1.0)

        for vu, cu, vn, cn in zip(self.v_updates, self.c_updates, self.v_norms, self.c_norms):
            c_to_v = torch.einsum("mn,bmh->bnh", h, c) / dv[None, :, None]
            v_to_c = torch.einsum("mn,bnh->bmh", h, v) / dc[None, :, None]
            v = vn(v + self.dropout(vu(torch.cat([v, c_to_v], dim=-1))))
            c = cn(c + self.dropout(cu(torch.cat([c, v_to_c], dim=-1))))

        bit_logits = self.bit_head(v).squeeze(-1)

        # Top-k suspicious tokens plus global token.
        bsz, n, hid = v.shape
        k = min(self.top_k_tokens, n)
        idx = heuristic_order[:, :k].long().clamp(0, n - 1)
        gather_idx = idx[:, :, None].expand(bsz, k, hid)
        tokens = torch.gather(v, 1, gather_idx)
        tokens = self.token_encoder(tokens)

        mean_pool = v.mean(dim=1)
        max_pool = v.max(dim=1).values
        token_pool = tokens.mean(dim=1)
        g = self.global_in(global_features)
        packet = self.packet_proj(torch.cat([mean_pool + g, max_pool, token_pool], dim=-1))

        return {
            "bit_logits": bit_logits,
            "segment_logits": self.segment_head(packet),
            "weight_logits": self.weight_head(packet),
            "standard_logits": self.standard_head(packet).squeeze(-1),
            "expanded_logits": self.expanded_head(packet).squeeze(-1),
            "rescue_logits": self.rescue_head(packet).squeeze(-1),
            "packet_embedding": packet,
            "candidate_scores": torch.empty((var_features.shape[0], 0), device=var_features.device),
        }
