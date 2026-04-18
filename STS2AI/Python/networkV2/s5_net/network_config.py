"""网络层数配置。

目的：支持 slim / full 两档预设，训练速度和模型能力之间做 trade-off。
后期想加回完整容量时不需要改代码，切换预设即可。

======================================================================
 重要：checkpoint 兼容性
======================================================================

不同配置的 checkpoint **不直接兼容**（层数变化 → 参数 shape 不同）。

但可以用 `load_compatible_params()`（见 UnifiedNet）做部分继承：
  - slim 版训好 → 把 shared 层（tokenizer / policy_head / decision_core 第1层
    / build_encoder）加载到 full 版
  - full 版新增的层（board_encoder 第2-3层 等）用零/小值初始化 + residual，
    fine-tune 时能在 slim 版基础上继续学习

典型工作流：
  1. 用 slim 预设训 N 轮（~5-10x 速度）→ checkpoint_slim.pt
  2. 改用 full 预设 + load_compatible_params(checkpoint_slim.pt) fine-tune
  3. 省 60-70% 从零训练的时间

======================================================================
 能力影响（参考 networkV2Final.md §6 的 attention 拆解）
======================================================================

slim 相对 full 的能力差距：
  BoardEncoder        3→1 层   小   （短序列影响不大）
  MechanismEncoder    2→1 层   小
  ModifierEncoder     2→1 层   小
  TurnPrefixEncoder   2→1 层   小-中 （牌序推理深度降低）
  CombatMemoryEncoder 1→1 层   无变化
  BuildSlots          3→1 iter 小
  ActionContextualizer 6→2 块  中   （6 段分别咨询 vs 合并查询，需要更多数据）
  DecisionCore        3→1 层   中   （动作互相比较深度降低，复杂权衡受影响）

综合：slim 需要 1.5-2x 样本达到 full 同水平，但训练速度快 ~2x，
     所以总训练时间基本持平，但早期反馈快、调参成本低。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NetworkConfig:
    """UnifiedNet 的层数 / 结构配置。

    所有字段都有默认值（对应 full 版 networkV2Final.md 的设计）。
    通过 preset() 工厂函数可以获得 slim / full / tiny 等预设。
    """

    # ---- 共享维度 ----
    d_model: int = 384
    n_heads: int = 8
    dropout: float = 0.1
    # 58:56 → 58 加 2 维 skada path prior(frequency + efficiency),
    # 来自 build_path_priors.py 的离线聚合表,data-driven 路径先验。
    max_numeric_dim: int = 58

    # ---- Memory Encoders (Layer 2) ----
    # 每个 encoder 的 self-attention 层数
    board_n_layers: int = 3
    mechanism_n_layers: int = 2
    modifier_n_layers: int = 2
    power_n_layers: int = 1          # v2: power_bank 编码层数（每个 active power 一个 token）
    prefix_n_layers: int = 2
    combat_memory_n_layers: int = 1

    # Build memory slot attention
    n_build_slots: int = 8
    build_n_iters: int = 3  # slot attention 迭代次数

    # ---- Action Contextualizer (Layer 3) ----
    # 当前设计是 6 段 cross-attention:
    #   并行 3 段 (board, modifier, mechanism) + 串行 3 段 (prefix, combat_memory, build)
    # contextualizer_mode:
    #   "full"  = 原版 6 段独立（最强，最慢）
    #   "merged" = 3 段并行合并成 1 cross + 3 段串行合并成 1 cross（快，能力略降）
    #   "minimal" = 全部 concat 成 1 个 kv 做单次 cross（最快，能力最弱）
    contextualizer_mode: str = "full"  # full / merged / minimal

    # ---- Decision Core (Layer 4) ----
    decision_n_layers: int = 3

    # ---- Option Contextualizer (Layer 3b, non-combat) ----
    # 同上，non-combat 分支
    option_contextualizer_mode: str = "full"

    # ---- Encounter Conditioning (方案 A: Conditional Policy) ----
    # 给 decision_repr 注入 boss/encounter-specific embedding，
    # 让 policy / value heads 自然条件化到当前对手，缓解多 boss 共享参数的梯度冲突。
    # vocab size 取 GAME_CATALOG 里 ~88 个 encounter，留 128 余量（新 DLC 扩展）。
    enable_encounter_conditioning: bool = True
    n_encounters: int = 128

    @property
    def is_slim(self) -> bool:
        """是否属于 slim 档（任一关键层被削减）。"""
        return (
            self.contextualizer_mode != "full"
            or self.decision_n_layers < 3
            or self.board_n_layers < 3
        )


# ======================================================================
# 预设
# ======================================================================

def preset_full() -> NetworkConfig:
    """完整主力版（networkV2Final.md 原始设计）。

    速度：~13-20ms/forward
    能力：最强
    参数：~36M (d_model=384)
    """
    return NetworkConfig(
        d_model=384, n_heads=8,
        board_n_layers=3,
        mechanism_n_layers=2, modifier_n_layers=2,
        prefix_n_layers=2, combat_memory_n_layers=1,
        build_n_iters=3,
        contextualizer_mode="full",
        option_contextualizer_mode="full",
        decision_n_layers=3,
    )


def preset_slim() -> NetworkConfig:
    """精简训练版。

    速度：~5-8ms/forward（省 attention ops: 22 → 7-8）
    能力：~80-90% of full（足够早期 PPO 训练）
    参数：~20M (d_model=384)
    用途：快速迭代、调参、benchmark。

    后期可用 load_compatible_params() 切回 full 版继续 fine-tune。
    """
    return NetworkConfig(
        d_model=384, n_heads=8,
        board_n_layers=1,
        mechanism_n_layers=1, modifier_n_layers=1,
        prefix_n_layers=1, combat_memory_n_layers=1,
        build_n_iters=1,
        contextualizer_mode="merged",
        option_contextualizer_mode="merged",
        decision_n_layers=1,
    )


def preset_tiny() -> NetworkConfig:
    """极简验证版（单元测试/调试用）。

    速度：~2-3ms/forward
    能力：最弱，仅验证 pipeline 正确
    参数：~4M (d_model=128)
    """
    return NetworkConfig(
        d_model=128, n_heads=4,
        board_n_layers=1,
        mechanism_n_layers=1, modifier_n_layers=1,
        prefix_n_layers=1, combat_memory_n_layers=1,
        n_build_slots=4, build_n_iters=1,
        contextualizer_mode="minimal",
        option_contextualizer_mode="minimal",
        decision_n_layers=1,
    )


PRESETS = {
    "full": preset_full,
    "slim": preset_slim,
    "tiny": preset_tiny,
}


def from_preset(name: str) -> NetworkConfig:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(PRESETS)}")
    return PRESETS[name]()
