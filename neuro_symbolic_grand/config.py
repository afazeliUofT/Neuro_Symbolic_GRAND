from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .utils.io import read_yaml


@dataclass
class CodeConfig:
    n: int = 128
    k: int = 64
    p_column_weight: int = 3
    seed: int = 1234


@dataclass
class ChannelConfig:
    backend: str = "sionna_tdl"
    allow_fallback: bool = True
    profile_family: str = "TDL"
    train_profiles: List[str] = field(default_factory=lambda: ["A", "C", "E"])
    eval_profiles: List[str] = field(default_factory=lambda: ["A", "C", "E"])
    delay_spread_s: float = 100e-9
    carrier_frequency_hz: float = 3.5e9
    subcarrier_spacing_hz: float = 15e3
    min_speed_mps: float = 0.0
    max_speed_mps: float = 10.0
    normalization: bool = True


@dataclass
class DataConfig:
    train_samples: int = 50000
    val_samples: int = 10000
    shard_size: int = 4000
    batch_size: int = 256
    snr_min_db: float = -4.0
    snr_max_db: float = 8.0
    trace_fraction: float = 0.01


@dataclass
class ModelConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ff_multiplier: int = 4
    dropout: float = 0.1
    num_segments: int = 8
    max_weight_class: int = 4


@dataclass
class TrainConfig:
    epochs: int = 12
    batch_size: int = 512
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 3
    bit_loss_weight: float = 1.0
    weight_loss_weight: float = 0.4
    segment_loss_weight: float = 0.3
    confidence_loss_weight: float = 0.2
    positive_bit_class_weight: float = 12.0
    label_smoothing: float = 0.0


@dataclass
class SearchConfig:
    baseline_pool_size: int = 24
    baseline_max_weight: int = 4
    baseline_budget: int = 2500
    baseline_weight_penalties: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.15, 0.4, 0.9])
    ai_pool_size: int = 18
    ai_max_weight: int = 4
    ai_budget: int = 1200
    fallback_budget: int = 1800
    ai_weight_penalties: List[float] = field(default_factory=lambda: [0.0, -0.05, 0.0, 0.2, 0.55])
    top_segments: int = 3
    top_bits_extra: int = 6
    candidate_mass_threshold: float = 0.85
    confidence_threshold: float = 0.35
    trace_top_attempts: int = 25


@dataclass
class EvalConfig:
    snr_grid_db: List[float] = field(default_factory=lambda: [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0])
    samples_per_point: int = 2000
    batch_size: int = 200
    trace_fraction: float = 0.02


@dataclass
class AnalysisConfig:
    top_trace_examples: int = 20


@dataclass
class ResourceConfig:
    generation_workers: int = 16
    generation_threads_per_worker: int = 4
    evaluation_workers: int = 16
    evaluation_threads_per_worker: int = 4
    train_torch_threads: int = 48
    train_torch_interop_threads: int = 1
    train_loader_workers: int = 4


@dataclass
class ExperimentConfig:
    experiment_name: str = "nsgrand_tdl_research"
    seed: int = 20260411
    output_dir: str = "outputs/nsgrand_tdl_research"
    code: CodeConfig = field(default_factory=CodeConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _update_dataclass(dc_obj: Any, values: Dict[str, Any]) -> Any:
    for key, value in values.items():
        current = getattr(dc_obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(dc_obj, key, value)
    return dc_obj


def load_config(config_path: str | Path | None = None) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if config_path is None:
        return cfg
    raw = read_yaml(Path(config_path))
    _update_dataclass(cfg, raw)
    return cfg
