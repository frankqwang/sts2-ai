"""辅助 head 监督 target 的计算。

历史问题：原来 train_full_run_v2.py 里直接把 RunBuildMemory 字段做线性组合作为 target，
等于让网络学输出 = 输入的恒等映射，毫无监督信息（white-train heads）。

本模块做两件事：
1. 把 target 计算从 train 主循环里抽出来，便于独立测试和迭代
2. 用 **多信号 + 非线性 + 未来导向** 重写每个 target，让它真正提供学习信号

不是 true expected value（那需要写 deck-vs-boss combat sim），但比原 heuristic
多出"floor pressure × build identity × resource"的非平凡函数关系。

后续可以接：
- forward_simulation.py 模拟 N 场 baseline 战斗给 deck_quality 真值
- boss_priors.py 不同 boss 对 build 类型的偏好
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from networkV2.s1_schema.memory import RunBuildMemory


# ---------------------------------------------------------------------------
# Floor pressure: 楼层越深，对 build/资源的要求越高
# ---------------------------------------------------------------------------

def floor_pressure(floor: int, act: int = 1) -> float:
    """返回 [0, 1] 的"楼层压力"系数。

    Act 1: 第 1 层 0.0 → 第 17 层（boss 前）0.9 → boss 1.0
    Act 2: 起步就 0.5，boss 前 0.9
    Act 3: 起步 0.7
    """
    base = {1: 0.0, 2: 0.5, 3: 0.7, 4: 1.0}.get(act, 0.0)
    floor_in_act = max(0, floor - {1: 0, 2: 17, 3: 34, 4: 51}.get(act, 0))
    progress = min(floor_in_act / 17.0, 1.0)  # 0 ~ 1
    return min(1.0, base + (1.0 - base) * progress)


# ---------------------------------------------------------------------------
# Build identity: 根据 build profile 判定 archetype，给予 archetype-specific 评分
# ---------------------------------------------------------------------------

@dataclass
class BuildArchetype:
    name: str
    quality: float    # [-1, 1] 该 archetype 的相对强度
    confidence: float  # [0, 1] 是否清晰属于该 archetype


def detect_archetype(rbm: RunBuildMemory) -> BuildArchetype:
    """用多信号判定 build archetype。

    规则：
    - frontload 高 + draw 中等 → 'aggro'（前期爆发，怕长战）
    - block 高 + scaling 中等 → 'turtle_scaling'（防守反击）
    - scaling 高 + draw 高 → 'scaling_engine'（长战发动机）
    - aoe 高 + frontload 中等 → 'aoe_clear'（清场流）
    - heal 高 → 'sustain'（续航流）
    - 都不突出 → 'undefined'（白板，差）

    每个 archetype 在 STS2 里强度不同（基于经验）：
    - scaling_engine: 0.7（boss 杀手）
    - aoe_clear: 0.6（act 2/3 elite/boss 多 add）
    - turtle_scaling: 0.5（稳）
    - aggro: 0.3（act 1 强 act 3 弱）
    - sustain: 0.2（性价比低）
    - undefined: -0.5（无方向）
    """
    fl, blk, drw, scl, aoe, hl = (
        rbm.frontload, rbm.block, rbm.draw,
        rbm.scaling, rbm.aoe, rbm.heal,
    )
    # 阈值（经验值，基于 build_profile 通常归一到 [0,1]）
    HI = 0.5
    MID = 0.3

    # 按强度排序判定
    if scl >= HI and drw >= MID:
        return BuildArchetype("scaling_engine", quality=0.7, confidence=min((scl + drw) / 2, 1.0))
    if aoe >= HI and fl >= MID:
        return BuildArchetype("aoe_clear", quality=0.6, confidence=min((aoe + fl) / 2, 1.0))
    if blk >= HI and scl >= MID:
        return BuildArchetype("turtle_scaling", quality=0.5, confidence=min((blk + scl) / 2, 1.0))
    if fl >= HI and drw >= MID:
        return BuildArchetype("aggro", quality=0.3, confidence=min((fl + drw) / 2, 1.0))
    if hl >= HI:
        return BuildArchetype("sustain", quality=0.2, confidence=hl)
    return BuildArchetype("undefined", quality=-0.5, confidence=0.5)


# ---------------------------------------------------------------------------
# 各个 head 的 target 计算
# ---------------------------------------------------------------------------

def compute_deck_quality_target(rbm: RunBuildMemory, hp_ratio: float = 1.0) -> float:
    """deck_quality_target ∈ [-1, 1]。

    包含 4 个独立分量：
      1. archetype quality × confidence （build identity）
      2. consistency - curse_density - 0.5 * high_cost_density （手感）
      3. damage potential = (frontload + scaling/2) * 0.5  （输出能力）
      4. defense potential = (block + heal/2) * 0.4 （防御能力）

    用 floor_pressure 加权 1/3/4，让靠后楼层对 quality 要求更高。
    最后 tanh 压到 [-1, 1] 防溢出。
    """
    arch = detect_archetype(rbm)
    fp = floor_pressure(rbm.floor, rbm.act)

    arch_score = arch.quality * arch.confidence
    deck_health = rbm.consistency - rbm.curse_density - 0.5 * rbm.high_cost_density
    damage_pot = (rbm.frontload + 0.5 * rbm.scaling) * 0.5
    defense_pot = (rbm.block + 0.5 * rbm.heal) * 0.4

    # floor pressure 越高，arch + damage + defense 的权重越大（要求 build 成型）
    combined = (
        arch_score * (0.4 + 0.4 * fp)
        + deck_health * 0.3
        + damage_pot * (0.2 + 0.3 * fp)
        + defense_pot * (0.2 + 0.3 * fp)
    )
    # HP 健康度作为修正：HP 低则 quality 整体打折（资源耗损）
    combined *= (0.7 + 0.3 * hp_ratio)
    return math.tanh(combined)


def compute_boss_readiness_target(rbm: RunBuildMemory, hp_ratio: float = 1.0) -> float:
    """boss_readiness_target ∈ [0, 1]。

    与 deck_quality 不同：boss 前要求**专项**成型（明确的高输出 OR 高防御），
    不是泛泛的 build quality。

    分量：
      - 输出充足度：frontload + scaling 的"sigmoid 阈值"（达到 0.6 才算够）
      - 防御充足度：block + heal*0.5 的 sigmoid
      - HP buffer
      - 资源储备（药水 + 遗物 hint）
    """
    fp = floor_pressure(rbm.floor, rbm.act)

    # 阶梯：低于 0.4 几乎 0；0.6 是中等；0.9+ 是 ready
    def _sigmoid_step(x: float, mid: float = 0.6, sharpness: float = 8.0) -> float:
        return 1.0 / (1.0 + math.exp(-sharpness * (x - mid)))

    damage_ready = _sigmoid_step(rbm.frontload + rbm.scaling)
    defense_ready = _sigmoid_step(rbm.block + 0.5 * rbm.heal, mid=0.5)
    # boss 前要么输出够，要么防御够（取 max 而不是 sum：不需要全能）
    spec_ready = max(damage_ready, defense_ready)

    hp_buffer = hp_ratio  # [0,1]
    resource = min(rbm.potion_count / 3.0, 1.0) * 0.3 + min(rbm.relic_count / 8.0, 1.0) * 0.2

    # boss 前的 readiness：spec_ready 是主项，hp+resource 是修饰
    raw = 0.6 * spec_ready + 0.25 * hp_buffer + 0.15 * resource
    # floor 越靠近 boss，要求越严格（负偏置）
    if fp >= 0.7:
        raw -= 0.1 * (fp - 0.7) / 0.3  # boss 临近时降低基线
    return max(0.0, min(1.0, raw))


def compute_resource_health_target(rbm: RunBuildMemory, hp_ratio: float = 1.0) -> float:
    """resource_health_target ∈ [0, 1]。

    资源 = HP + 药水 + 金币 + （curse-free 的 deck）
    晚期 floor 对资源的要求相对降低（既然到了说明前期没崩）。
    """
    hp_score = hp_ratio
    potion_score = min(rbm.potion_count / 3.0, 1.0)
    gold_score = min(rbm.gold / 200.0, 1.0)  # 200g 算手头宽裕
    deck_clean = max(0.0, 1.0 - 2.0 * rbm.curse_density)

    return max(0.0, min(1.0,
        0.4 * hp_score + 0.2 * potion_score + 0.2 * gold_score + 0.2 * deck_clean
    ))


def compute_resource_retention_target(rbm: RunBuildMemory, hp_ratio: float = 1.0) -> float:
    """resource_retention_target ∈ [0, 1]。

    与 resource_health 区别：retention 关注"还剩多少 / 用了多少"，
    health 关注"绝对充足度"。
    """
    hp_retain = hp_ratio
    potion_used_ratio = min(rbm.potions_used_total / max(rbm.potions_used_total + rbm.potion_count, 1), 1.0)
    potion_retain = 1.0 - potion_used_ratio
    # 战斗损耗效率：经历的战斗多 + HP 高 = retention 好
    combat_efficiency = 0.0
    if rbm.combats_seen > 0:
        avg_hp_loss = rbm.total_hp_lost / rbm.combats_seen
        # 平均每场掉 < 5 = 1.0；> 25 = 0.0
        combat_efficiency = max(0.0, min(1.0, 1.0 - (avg_hp_loss - 5) / 20))

    return max(0.0, min(1.0,
        0.5 * hp_retain + 0.2 * potion_retain + 0.3 * combat_efficiency
    ))


__all__ = [
    "floor_pressure",
    "detect_archetype",
    "BuildArchetype",
    "compute_deck_quality_target",
    "compute_boss_readiness_target",
    "compute_resource_health_target",
    "compute_resource_retention_target",
]
