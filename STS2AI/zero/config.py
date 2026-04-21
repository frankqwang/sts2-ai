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
class CollectConfig:
    """Collector 行为配置。

    `episodes_per_iteration` 只控制每轮收集多少局，和评估阶段的
    `episodes_per_cohort` 分开，避免两者共用一个字段产生歧义。
    """

    episodes_per_iteration: int = 32
    max_steps_per_episode: int = 200
    epsilon_greedy: float = 0.0
    temperature: float = 0.0
    mode: str = "search_only_collect"
    search_guidance_priority_threshold: float = 1.2
    search_guidance_max_steps_per_episode: int = 8
    search_guidance_target_encounters: tuple[str, ...] = ()
    search_guidance_port_offset: int = 100


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
    id_hash_buckets: int = 8192


@dataclass(slots=True)
class LossWeights:
    policy: float = 1.0
    value: float = 0.5
    ranking: float = 0.5
    delta: float = 0.2
    uncertainty: float = 0.1
    policy_search_kl_weight: float = 1.25
    policy_behavior_ce_weight: float = 0.75
    policy_bad_rollout_ce_scale: float = 0.35


@dataclass(slots=True)
class PoolConfig:
    recent_online_weight: float = 0.35
    search_weight: float = 0.25
    rare_weight: float = 0.20
    reanalyse_weight: float = 0.10
    legacy_weight: float = 0.10
    bucket_capacity: int = 2048
    rare_bucket_capacity: int = 256
    search_bucket_capacity: int = 1024
    """样本池默认权重与基础容量。

    基础容量只是冷启动下限。运行中会根据最近若干轮样本量动态放大，
    目标是让总池容量至少覆盖最近两轮的逻辑样本总量，再在此基础上按
    `keep_score` 做留优淘汰。
    """

    dynamic_capacity_enabled: bool = True
    dynamic_capacity_recent_iterations: int = 2


@dataclass(slots=True)
class SearchConfig:
    mode: str = "weak"
    top2_gap_threshold: float = 0.05
    uncertainty_threshold: float = 0.55
    near_lethal_hp_ratio: float = 0.25
    max_requests_per_iteration: int = 512
    max_root_actions: int = 8
    rollouts_per_action: int = 2
    max_branch_steps: int = 24
    allow_branching: bool = False
    rollout_policy: str = "aggregate_search_prior"
    rollout_scorer: str = "fight_quality"
    trace_topk: int = 4
    enable_snapshot_restore: bool = True
    leaf_eval_horizon: int = 3
    leaf_value_weight: float = 0.75
    root_cache_size: int = 128


@dataclass(slots=True)
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    steps_per_iteration: int = 200
    grad_clip_norm: float = 1.0
    warmup_steps: int = 32
    min_lr_ratio: float = 0.10
    device: str = "auto"
    amp_enabled: bool = True
    prefetch_batches: int = 2


@dataclass(slots=True)
class EvalConfig:
    episodes_per_cohort: int = 32
    promote_min_win_rate_gain: float = 0.01
    allow_hp_quality_drop: float = 0.02
    promote_min_search_agreement_gain: float = 0.0
    promote_min_enemy_hp_gain: float = 0.0
    promote_min_fight_quality_gain: float = 0.0
    significance_z: float = 0.0
    max_timeout_rate: float = 0.0
    max_no_progress_ratio: float = 0.95
    max_no_progress_streak: float = 128.0


@dataclass(slots=True)
class CheckpointConfig:
    keep_rejected_checkpoints: bool = False
    active_pointer_name: str = "active.json"


@dataclass(slots=True)
class ZeroConfig:
    seed: int = 20260420
    paths: ZeroPaths = field(default_factory=ZeroPaths)
    collect: CollectConfig = field(default_factory=CollectConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    losses: LossWeights = field(default_factory=LossWeights)
    pools: PoolConfig = field(default_factory=PoolConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)


ZERO_RUNTIME_DEFAULTS = RuntimeDefaults()
