from __future__ import annotations

from dataclasses import dataclass, field

from .paths import ZeroPaths


@dataclass(slots=True)
class RuntimeDefaults:
    """Zero 侧运行入口的默认参数。

    这里允许保留少量“入口默认值”，方便 CLI / smoke / replay 工具直接运行，
    但它们不是游戏权威数据源。真正的卡牌/遗物/角色/build 数据必须继续从
    game_wiki sqlite 或 runtime API 读取，不要在业务逻辑里散落硬编码。
    """

    default_character_id: str = "IRONCLAD"
    default_port: int = 15527
    default_connect_timeout_s: float = 15.0


@dataclass(slots=True)
class EncoderConfig:
    hidden_dim: int = 256
    action_dim: int = 192
    history_dim: int = 256
    history_steps: int = 8
    history_layers: int = 4
    history_heads: int = 4
    max_enemies: int = 4
    max_hand_cards: int = 10
    buff_slots: int = 16
    id_hash_buckets: int = 2048


@dataclass(slots=True)
class LossWeights:
    policy: float = 1.0
    value: float = 0.5
    ranking: float = 0.5
    delta: float = 0.2
    uncertainty: float = 0.1


@dataclass(slots=True)
class PoolConfig:
    recent_online_weight: float = 0.35
    teacher_weight: float = 0.25
    rare_weight: float = 0.20
    reanalyse_weight: float = 0.10
    legacy_weight: float = 0.10
    bucket_capacity: int = 2048
    rare_bucket_capacity: int = 256
    teacher_bucket_capacity: int = 1024


@dataclass(slots=True)
class TeacherConfig:
    top2_gap_threshold: float = 0.05
    uncertainty_threshold: float = 0.55
    near_lethal_hp_ratio: float = 0.25
    max_requests_per_iteration: int = 512


@dataclass(slots=True)
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    steps_per_iteration: int = 200
    grad_clip_norm: float = 1.0


@dataclass(slots=True)
class EvalConfig:
    episodes_per_cohort: int = 32
    promote_min_win_rate_gain: float = 0.01
    allow_hp_remaining_drop: float = 0.02


@dataclass(slots=True)
class ZeroConfig:
    paths: ZeroPaths = field(default_factory=ZeroPaths)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    losses: LossWeights = field(default_factory=LossWeights)
    pools: PoolConfig = field(default_factory=PoolConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)


ZERO_RUNTIME_DEFAULTS = RuntimeDefaults()
