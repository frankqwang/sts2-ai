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
    final_epsilon_greedy: float | None = None
    final_temperature: float | None = None
    anneal_iterations: int = 1

    def resolve_for_iteration(self, iteration: int) -> tuple[float, float]:
        epsilon_end = (
            float(self.final_epsilon_greedy)
            if self.final_epsilon_greedy is not None
            else float(self.epsilon_greedy)
        )
        temperature_end = (
            float(self.final_temperature)
            if self.final_temperature is not None
            else float(self.temperature)
        )
        total = max(1, int(self.anneal_iterations or 1))
        if total <= 1:
            progress = 1.0
        else:
            progress = min(max(float(iteration - 1), 0.0), float(total - 1)) / float(total - 1)
        epsilon = float(self.epsilon_greedy) + (epsilon_end - float(self.epsilon_greedy)) * progress
        temperature = float(self.temperature) + (temperature_end - float(self.temperature)) * progress
        return max(0.0, epsilon), max(0.0, temperature)


@dataclass(slots=True)
class EncoderConfig:
    policy_arch: str = "flat"
    history_variant: str = "recurrent_gru"
    model_variant: str | None = None
    hidden_dim: int = 256
    action_dim: int = 192
    future_summary_dim: int = 3
    history_dim: int = 256
    history_steps: int = 8
    history_layers: int = 1
    history_heads: int = 4
    history_dropout: float = 0.1
    history_gate_bias: float = -1.5
    token_backbone_layers: int = 4
    token_backbone_heads: int = 4
    intent_vocab_size: int = 4
    intent_dim: int = 64
    microbatch_window_ms: float = 2.0
    max_enemies: int = 4
    max_hand_cards: int = 10
    buff_slots: int = 16
    id_hash_buckets: int = 8192


@dataclass(slots=True)
class LossWeights:
    policy: float = 1.0
    policy_align: float = 0.25
    value: float = 0.5
    delta: float = 0.2
    future_summary: float = 0.1
    submenu_policy: float = 0.35
    submenu_confirm: float = 0.2
    policy_behavior_ce_weight: float = 1.0
    policy_value_align_temperature: float = 0.75


@dataclass(slots=True)
class PoolConfig:
    recent_online_weight: float = 0.45
    rare_weight: float = 0.25
    reanalyse_weight: float = 0.15
    legacy_weight: float = 0.15
    bucket_capacity: int = 2048
    rare_bucket_capacity: int = 256
    """样本池默认权重与基础容量。

    基础容量只是冷启动下限。运行中会根据最近若干轮样本量动态放大，
    目标是让总池容量至少覆盖最近两轮的逻辑样本总量，再在此基础上按
    `keep_score` 做留优淘汰。
    """

    dynamic_capacity_enabled: bool = True
    dynamic_capacity_recent_iterations: int = 2


@dataclass(slots=True)
class TrainConfig:
    algorithm: str = "behavior_clone"
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
    ppo_clip_ratio: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_turn_intent_coef: float = 1.0
    ppo_turn_value_coef: float = 0.5
    ppo_future_summary_coef: float = 0.10
    ppo_policy_align_coef: float = 0.15
    ppo_submenu_policy_coef: float = 0.35
    ppo_submenu_confirm_coef: float = 0.20
    ppo_policy_align_temperature: float = 0.75
    ppo_behavior_ce_coef: float = 0.10
    ppo_action_entropy_coef: float = 0.01
    ppo_intent_entropy_coef: float = 0.05
    ppo_entropy_coef: float = 0.01
    action_imitation_adv_temperature: float = 5.0
    ppo_advantage_norm: bool = True
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_epochs: int = 4
    ppo_value_clip: float = 0.2
    normalize_step_returns: bool = True
    normalize_turn_returns: bool = True


@dataclass(slots=True)
class EvalConfig:
    episodes_per_cohort: int = 32
    promote_min_win_rate_gain: float = 0.01
    allow_hp_quality_drop: float = 0.02
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
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)


ZERO_RUNTIME_DEFAULTS = RuntimeDefaults()
