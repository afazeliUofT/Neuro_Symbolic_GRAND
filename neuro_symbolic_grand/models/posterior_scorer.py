from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn


class MeanPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


class PosteriorSchedulerNet(nn.Module):
    def __init__(
        self,
        n_bits: int,
        per_bit_dim: int,
        global_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_multiplier: int,
        dropout: float,
        num_segments: int,
        max_weight_class: int,
    ) -> None:
        super().__init__()
        self.n_bits = int(n_bits)
        self.num_segments = int(num_segments)
        self.max_weight_class = int(max_weight_class)
        self.bit_proj = nn.Linear(per_bit_dim, d_model)
        self.global_proj = nn.Linear(global_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_bits, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_multiplier * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = MeanPool()
        self.bit_head = nn.Linear(d_model, 1)
        self.segment_head = nn.Linear(d_model, num_segments)
        self.weight_head = nn.Linear(d_model, max_weight_class + 2)  # last class => > max_weight_class
        self.confidence_head = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(self, bit_features: torch.Tensor, global_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        global_token = self.global_proj(global_features).unsqueeze(1)
        x = self.bit_proj(bit_features) + self.pos_embedding + global_token
        x = self.encoder(x)
        x = self.norm(self.dropout(x))
        pooled = self.pool(x)
        return {
            "bit_logits": self.bit_head(x).squeeze(-1),
            "segment_logits": self.segment_head(pooled),
            "weight_logits": self.weight_head(pooled),
            "confidence_logit": self.confidence_head(pooled).squeeze(-1),
        }


@dataclass
class InferenceOutputs:
    bit_flip_prob: torch.Tensor
    segment_prob: torch.Tensor
    weight_prob: torch.Tensor
    confidence_prob: torch.Tensor


@torch.no_grad()
def run_model_inference(model: PosteriorSchedulerNet, bit_features: torch.Tensor, global_features: torch.Tensor) -> InferenceOutputs:
    model.eval()
    outputs = model(bit_features, global_features)
    return InferenceOutputs(
        bit_flip_prob=torch.sigmoid(outputs["bit_logits"]),
        segment_prob=torch.sigmoid(outputs["segment_logits"]),
        weight_prob=torch.softmax(outputs["weight_logits"], dim=-1),
        confidence_prob=torch.sigmoid(outputs["confidence_logit"]),
    )
