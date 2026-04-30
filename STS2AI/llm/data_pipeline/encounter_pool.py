"""Skada combat reset case loader.

- encounter_id 必须是 sim 能识别的大写形式（GAME_CATALOG 里的是小写，reset 接口用大写）
- 训练/评估入口只从 Skada cases.jsonl 构造 EncounterSpec
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EncounterSpec:
    encounter_id: str
    build: dict[str, Any] = field(default_factory=dict)
    tag: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def build_fingerprint(build: dict[str, Any]) -> str:
    """Stable short id for a combat build."""
    payload = json.dumps(build or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def encounter_key(spec: EncounterSpec) -> str:
    """Unique key for reward grouping and eval parity.

    The same encounter with different deck/relic builds is a different task.
    Grouping only by encounter_id gives noisy or wrong advantage signals.
    """
    tag = spec.tag or "untagged"
    return f"{spec.encounter_id}::{tag}::{build_fingerprint(spec.build)}"


def encounter_label(spec: EncounterSpec) -> str:
    tag = spec.tag or "untagged"
    return f"{spec.encounter_id}[{tag}:{build_fingerprint(spec.build)}]"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _act_from_floor(floor: int) -> int:
    if 1 <= floor <= 17:
        return 1
    if 18 <= floor <= 34:
        return 2
    if 35 <= floor <= 51:
        return 3
    if floor >= 52:
        return 4
    return 0


def _copy_build_with_floor(case: dict[str, Any]) -> dict[str, Any]:
    build = dict(case.get("build") if isinstance(case.get("build"), dict) else {})
    floor = _safe_int(case.get("floor"), 0)
    if floor > 0:
        build["floor"] = floor
    return build


def skada_case_to_spec(case: dict[str, Any]) -> EncounterSpec:
    """Convert one Skada single-combat case to an EncounterSpec.

    The sim combat_reset schema does not carry floor separately, so floor is
    stored in build/metadata and later injected into rendered rollout states.
    """
    floor = _safe_int(case.get("floor"), 0)
    act = _act_from_floor(floor)
    encounter_type = str(case.get("encounter_type") or "unknown").lower()
    tag = f"skada_floor_{floor:02d}_{encounter_type}" if floor > 0 else f"skada_{encounter_type}"
    return EncounterSpec(
        str(case.get("encounter_id") or "").upper(),
        _copy_build_with_floor(case),
        tag,
        {
            "source": "skada_case",
            "case_id": case.get("case_id"),
            "source_path": case.get("source_path"),
            "source_line": case.get("source_line"),
            "run_id": case.get("run_id"),
            "seed": case.get("seed"),
            "character_id": case.get("character_id"),
            "ascension": case.get("ascension"),
            "act": act if act > 0 else None,
            "floor": floor,
            "encounter_type": case.get("encounter_type"),
            "won": case.get("won"),
            "floor_state": case.get("floor_state") if isinstance(case.get("floor_state"), dict) else {},
            "case_metadata": case.get("metadata") if isinstance(case.get("metadata"), dict) else {},
        },
    )


def _is_hold_out_case(case_id: str | None, hold_out_fraction: float, hold_out_seed: int) -> bool:
    """deterministic hash 判断某 case 是否在 hold-out（eval-only）集合里。

    用 sha1(case_id:seed) 切 bucket，hold_out_fraction 决定 hold-out 占比。
    跨轮稳定（同 case_id+seed 永远同结论），让 hold-out pool 跨训练轮次都不变。
    """
    if hold_out_fraction <= 0.0 or not case_id:
        return False
    digest = hashlib.sha1(f"{case_id}|{hold_out_seed}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10000  # 取前 32 bit 足够
    threshold = int(round(hold_out_fraction * 10000))
    return bucket < threshold


def load_skada_case_pool(
    case_index: str | Path,
    *,
    character_id: str = "IRONCLAD",
    floor_min: int = 1,
    floor_max: int = 17,
    won_only: bool = True,
    limit: int = 0,
    sample_seed: int = 0,
    sample_mode: str = "stratified",
    elite_oversample_ratio: float = 0.0,
    boss_oversample_ratio: float = 0.0,
    pool_role: str = "full",
    hold_out_fraction: float = 0.0,
    hold_out_seed: int = 20260101,
    archetype_min_counts: dict[str, int] | None = None,
) -> list[EncounterSpec]:
    """Load floor-aware Skada combat reset cases for rollout training.

    sample_mode:
        - ``"file"``: 顺序取前 ``limit`` 个 case（确定性，无随机）。
        - ``"random"``: 全集打乱后取前 ``limit`` 个（最大随机，可能集中在某些 floor / encounter）。
        - ``"stratified"``: 按 ``(floor, encounter_type)`` 分桶 round-robin 取，覆盖 floor / 战斗类型。
        - ``"diverse"``: 多维度覆盖采样，按以下优先级保证 case 集合的覆盖广度：
            1. 主分桶 ``encounter_id``（保证 38 种怪都有机会被选中）。
            2. 同 encounter 内按 ``build_signature``（relic 集 + 主要卡牌摘要）二级分桶，
               让同种怪在不同 build / floor 下都被见到。
            3. round-robin 取，限度内尽量覆盖更多 encounter / build。

    elite_oversample_ratio:
        强制 elite case 在最终 sample 里占比 ≥ 该值（0.0-1.0）。当 case_limit 很小时
        (e.g. 16)，原 stratified/random 容易选到 0-1 个 elite（因 elite 在源集只占 ~6%）。
        设 0.3 就强制至少 ``ceil(limit * 0.3)`` 个 elite。
    """
    path = Path(case_index)
    specs: list[EncounterSpec] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(case, dict):
                continue
            if str(case.get("character_id") or "").upper() != character_id.upper():
                continue
            floor = _safe_int(case.get("floor"), 0)
            if floor < floor_min or floor > floor_max:
                continue
            if won_only and case.get("won") is not True:
                continue
            encounter_id = str(case.get("encounter_id") or "").strip()
            build = case.get("build")
            if not encounter_id or not isinstance(build, dict):
                continue
            specs.append(skada_case_to_spec(case))

    # 按 pool_role 切分 train / eval（hold-out）。'full' 不切，'train' 排除 hold-out，
    # 'eval' 只取 hold-out。pool_role 不识别时按 'full' 处理。
    role = (pool_role or "full").lower()
    if role in {"train", "eval"} and hold_out_fraction > 0.0:
        def _case_id(spec: EncounterSpec) -> str | None:
            meta = spec.metadata if isinstance(spec.metadata, dict) else {}
            return str(meta.get("case_id") or "")
        specs = [
            s for s in specs
            if (
                _is_hold_out_case(_case_id(s), hold_out_fraction, hold_out_seed)
                if role == "eval"
                else not _is_hold_out_case(_case_id(s), hold_out_fraction, hold_out_seed)
            )
        ]

    if limit <= 0 or len(specs) <= limit:
        return specs
    mode = (sample_mode or "file").strip().lower()
    if mode == "file":
        return specs[:limit]
    rng = random.Random(sample_seed)
    if mode == "random":
        shuffled = list(specs)
        rng.shuffle(shuffled)
        return shuffled[:limit]
    if mode == "stratified":
        sampled = _stratified_case_sample(specs, limit=limit, rng=rng)
    elif mode == "diverse":
        sampled = _diverse_case_sample(specs, limit=limit, rng=rng)
    else:
        raise ValueError(f"unsupported Skada case sample_mode: {sample_mode}")
    # 先补 boss（最稀缺、机制最特殊），再补 elite，最后保 archetype 多样性
    if boss_oversample_ratio > 0.0:
        sampled = _enforce_boss_ratio(sampled, specs, ratio=boss_oversample_ratio, rng=rng)
    if elite_oversample_ratio > 0.0:
        sampled = _enforce_elite_ratio(sampled, specs, ratio=elite_oversample_ratio, rng=rng)
    if archetype_min_counts:
        sampled = _enforce_archetype_min_counts(sampled, specs, requirements=archetype_min_counts, rng=rng)
    return sampled


def _stratified_case_sample(specs: list[EncounterSpec], *, limit: int, rng: random.Random) -> list[EncounterSpec]:
    groups: dict[tuple[int, str], list[EncounterSpec]] = defaultdict(list)
    for spec in specs:
        meta = spec.metadata if isinstance(spec.metadata, dict) else {}
        floor = _safe_int(meta.get("floor"), 0)
        encounter_type = str(meta.get("encounter_type") or "unknown").lower()
        groups[(floor, encounter_type)].append(spec)

    buckets = sorted(groups)
    for bucket in buckets:
        rng.shuffle(groups[bucket])

    selected: list[EncounterSpec] = []
    cursor = 0
    while len(selected) < limit and buckets:
        bucket = buckets[cursor % len(buckets)]
        items = groups[bucket]
        if items:
            selected.append(items.pop())
        if not items:
            buckets.remove(bucket)
            if not buckets:
                break
            cursor %= len(buckets)
        else:
            cursor += 1
    return selected


def _build_signature(spec: EncounterSpec) -> str:
    """生成 build 签名：relic 集 + 卡组 archetype 摘要。

    用途：让 ``diverse`` 模式区分同 encounter 不同 build 的 case，
    避免反复选到同 relic / 同卡组的样本。
    签名是 deterministic 的字符串（不需要稳定性）；hash 后用作分桶 key 即可。
    """
    build = spec.build if isinstance(spec.build, dict) else {}
    relics = build.get("relics") or []
    relic_ids = []
    for r in relics:
        if isinstance(r, str):
            relic_ids.append(r)
        elif isinstance(r, dict):
            rid = r.get("id") or r.get("relic_id")
            if rid:
                relic_ids.append(str(rid))
    relic_sig = ",".join(sorted(relic_ids))
    deck = build.get("deck") or []
    deck_card_count: dict[str, int] = defaultdict(int)
    for c in deck:
        if isinstance(c, str):
            deck_card_count[c] += 1
        elif isinstance(c, dict):
            cid = c.get("id") or c.get("card_id")
            if cid:
                deck_card_count[str(cid)] += 1
    # 取卡组里出现次数最多的 5 张作为 archetype 代表（忽略起始 STRIKE/DEFEND 的固定多张噪声）
    top_cards = sorted(
        ((cid, n) for cid, n in deck_card_count.items() if cid not in {"STRIKE_IRONCLAD", "DEFEND_IRONCLAD"}),
        key=lambda kv: (-kv[1], kv[0]),
    )[:5]
    deck_sig = ",".join(f"{cid}x{n}" for cid, n in top_cards)
    return f"R[{relic_sig}]|D[{deck_sig}]"


def _diverse_case_sample(specs: list[EncounterSpec], *, limit: int, rng: random.Random) -> list[EncounterSpec]:
    """多维度覆盖采样。

    分桶层级：
        encounter_id -> build_signature -> [specs]

    取样策略：
        外层 round-robin 不同 encounter_id（保证 38 种怪覆盖广），
        内层每次从 build_signature 桶中也 round-robin（保证同怪不同 build 都见过）。
    """
    by_enc: dict[str, dict[str, list[EncounterSpec]]] = defaultdict(lambda: defaultdict(list))
    for spec in specs:
        eid = str(spec.encounter_id or "?")
        sig = _build_signature(spec)
        by_enc[eid][sig].append(spec)

    # 每个 encounter 内打乱 build_signature 顺序 + 同 sig 内打乱
    enc_order: list[str] = sorted(by_enc.keys())
    rng.shuffle(enc_order)
    enc_state: dict[str, list[list[EncounterSpec]]] = {}
    for eid in enc_order:
        sig_groups = list(by_enc[eid].values())
        rng.shuffle(sig_groups)
        for grp in sig_groups:
            rng.shuffle(grp)
        enc_state[eid] = sig_groups

    selected: list[EncounterSpec] = []
    cursor = 0
    active = list(enc_order)
    while len(selected) < limit and active:
        eid = active[cursor % len(active)]
        sig_groups = enc_state[eid]
        if not sig_groups:
            active.remove(eid)
            if not active:
                break
            cursor %= len(active)
            continue
        # 从该 encounter 的第一个非空 build sig 桶取一个
        grp = sig_groups[0]
        spec = grp.pop()
        selected.append(spec)
        if not grp:
            sig_groups.pop(0)
        else:
            # 把这个 sig 桶轮到队尾，下次让该 encounter 选不同 build
            sig_groups.append(sig_groups.pop(0))
        cursor += 1
    return selected


# 卡组 archetype 分类。每条目是 archetype name -> 该 archetype 的代表卡集合。
# 用于 archetype-aware sampling：按需要选含特定 archetype 卡的 case，让 model 见过
# 各种 build 风格而不是只见 strike/defend 主流。
# NOTE: 这只是分类用的"特征卡"，不是全集；选标志性卡能区分 archetype 即可。
_ARCHETYPE_FEATURE_CARDS: dict[str, set[str]] = {
    "multi_hit": {
        "TWIN_STRIKE", "POMMEL_STRIKE", "FIEND_FIRE", "DAGGER_SPRAY",
        "RAMPAGE", "PUMMEL", "DAGGER_THROW", "ONE_TWO_PUNCH", "GENETIC_ALGORITHM",
    },
    "aoe": {
        "THUNDERCLAP", "CLEAVE", "WHIRLWIND", "IMMOLATE", "CARNAGE",
        "SHOCKWAVE", "BREAKTHROUGH", "FLICK_FLACK", "ECHOING_SLASH",
        "DRAMATIC_ENTRANCE", "ASTRAL_PULSE", "BANSHEES_CRY",
    },
    "power_build": {
        "INFLAME", "DEMON_FORM", "BARRICADE", "JUGGERNAUT",
        "METALLICIZE", "RUPTURE", "FEEL_NO_PAIN", "DARK_EMBRACE",
        "FIRE_BREATHING", "EVOLVE", "AFTERIMAGE",
    },
    "lethal_burst": {
        "BLUDGEON", "FEED", "REAPER", "SEARING_BLOW", "SEVER_SOUL",
        "BODY_SLAM", "HEAVY_BLADE",
    },
    "exhaust": {
        "TRUE_GRIT", "OFFERING", "FIEND_FIRE", "SECOND_WIND", "DARK_EMBRACE",
        "FEEL_NO_PAIN", "CORRUPTION", "REAPER", "SEVER_SOUL",
    },
    "block_engine": {
        "ENTRENCH", "BARRICADE", "JUGGERNAUT", "METALLICIZE",
        "BODY_SLAM", "FLAME_BARRIER", "SHRUG_IT_OFF", "GHOSTLY_ARMOR",
    },
}


def classify_build_archetypes(spec: EncounterSpec) -> set[str]:
    """根据 build.deck 里的特征卡，判断该 case 涉及哪些 archetype。

    一个 build 可同时属于多个 archetype（如 INFLAME+RUPTURE+POMMEL_STRIKE 既是
    power_build 又有 multi_hit 元素）。
    """
    if not isinstance(spec.build, dict):
        return set()
    deck = spec.build.get("deck") or []
    deck_ids: set[str] = set()
    for c in deck:
        if isinstance(c, str):
            deck_ids.add(c.upper())
        elif isinstance(c, dict):
            cid = c.get("id") or c.get("card_id")
            if cid:
                deck_ids.add(str(cid).upper())
    archetypes: set[str] = set()
    for archetype, feature_cards in _ARCHETYPE_FEATURE_CARDS.items():
        if deck_ids & feature_cards:
            archetypes.add(archetype)
    return archetypes


def _spec_encounter_type(spec: EncounterSpec) -> str:
    meta = spec.metadata if isinstance(spec.metadata, dict) else {}
    return str(meta.get("encounter_type") or "").lower()


def _enforce_type_min_count(
    sampled: list[EncounterSpec],
    full_pool: list[EncounterSpec],
    *,
    target_type: str,
    min_count: int,
    rng: random.Random,
) -> list[EncounterSpec]:
    """通用：在 sampled 里保证 ``encounter_type == target_type`` 的至少 min_count 个。

    缺多少就从 full_pool 找补，把等量的 normal（既非 elite 也非 boss）替换掉。
    优先替换 normal 而非其它特殊类型，避免把已经 oversample 进来的 elite/boss 又挤掉。
    """
    if min_count <= 0 or not sampled:
        return sampled

    def _is_target(spec: EncounterSpec) -> bool:
        return _spec_encounter_type(spec) == target_type

    current = sum(1 for s in sampled if _is_target(s))
    deficit = min_count - current
    if deficit <= 0:
        return sampled

    sampled_ids = {str((s.metadata or {}).get("case_id")) for s in sampled}
    candidates = [
        s for s in full_pool
        if _is_target(s) and str((s.metadata or {}).get("case_id")) not in sampled_ids
    ]
    if not candidates:
        return sampled
    rng.shuffle(candidates)
    additions = candidates[:deficit]

    # 优先替换 normal（既不是 elite 也不是 boss）；不够再替换其它非 target 类型
    normals = [(i, s) for i, s in enumerate(sampled) if _spec_encounter_type(s) == "normal"]
    others = [(i, s) for i, s in enumerate(sampled) if _spec_encounter_type(s) not in ("normal", target_type)]
    rng.shuffle(normals)
    rng.shuffle(others)
    drop_pool = normals + others
    drop_indices = {idx for idx, _ in drop_pool[:len(additions)]}

    out: list[EncounterSpec] = []
    for i, s in enumerate(sampled):
        if i not in drop_indices:
            out.append(s)
    out.extend(additions)
    return out


def _enforce_elite_ratio(
    sampled: list[EncounterSpec],
    full_pool: list[EncounterSpec],
    *,
    ratio: float,
    rng: random.Random,
) -> list[EncounterSpec]:
    """elite 占比保底（保留向后兼容；底层调用 _enforce_type_min_count）。"""
    import math
    if ratio <= 0.0 or not sampled:
        return sampled
    min_count = max(0, math.ceil(len(sampled) * ratio))
    return _enforce_type_min_count(sampled, full_pool, target_type="elite", min_count=min_count, rng=rng)


def _enforce_boss_ratio(
    sampled: list[EncounterSpec],
    full_pool: list[EncounterSpec],
    *,
    ratio: float,
    rng: random.Random,
) -> list[EncounterSpec]:
    """boss 占比保底。boss 在源池占比极低（floor 17/33/48），不强制就采不到。"""
    import math
    if ratio <= 0.0 or not sampled:
        return sampled
    min_count = max(0, math.ceil(len(sampled) * ratio))
    return _enforce_type_min_count(sampled, full_pool, target_type="boss", min_count=min_count, rng=rng)


def _enforce_archetype_min_counts(
    sampled: list[EncounterSpec],
    full_pool: list[EncounterSpec],
    *,
    requirements: dict[str, int],
    rng: random.Random,
) -> list[EncounterSpec]:
    """让 sampled 里每个 requested archetype 至少有 ``min_count`` 个 case。

    思路：
      1. 算每 archetype 当前覆盖 case 数。
      2. 对每个 deficit 的 archetype，从 full_pool 里找有该 archetype 标签且未在 sampled
         的 case，随机选 deficit 个补进去。
      3. 同等数量从 sampled 内**最低 archetype 覆盖度**的 case 替换出去（保留高覆盖 case）。

    这样多 archetype 同时要求时，单个 case 可能同时满足多个 archetype（如 FIEND_FIRE 既
    是 multi_hit 又是 exhaust），换 1 张能补 2 个缺口。
    """
    if not requirements or not sampled:
        return sampled

    # 当前 sampled 里每 archetype 的 case 数
    sampled_archetypes: list[set[str]] = [classify_build_archetypes(s) for s in sampled]
    counts = {arch: 0 for arch in requirements}
    for archs in sampled_archetypes:
        for arch in archs & set(requirements.keys()):
            counts[arch] += 1

    deficit = {arch: max(0, need - counts[arch]) for arch, need in requirements.items()}
    if all(d == 0 for d in deficit.values()):
        return sampled  # 已全部满足

    sampled_ids = {str((s.metadata or {}).get("case_id")) for s in sampled}
    # 候选池：未在 sampled 里且至少满足一个 deficit archetype
    candidates: list[tuple[EncounterSpec, set[str]]] = []
    for s in full_pool:
        cid = str((s.metadata or {}).get("case_id"))
        if cid in sampled_ids:
            continue
        archs = classify_build_archetypes(s)
        useful = archs & {a for a, d in deficit.items() if d > 0}
        if useful:
            candidates.append((s, archs))
    rng.shuffle(candidates)

    additions: list[EncounterSpec] = []
    for cand, archs in candidates:
        useful = archs & {a for a, d in deficit.items() if d > 0}
        if not useful:
            continue
        additions.append(cand)
        for a in useful:
            deficit[a] = max(0, deficit[a] - 1)
        if all(d == 0 for d in deficit.values()):
            break

    if not additions:
        return sampled

    # 替换：移除 archetype 覆盖最少的 case（priority 给保留多 archetype 的）
    # 重新按"被请求 archetype 的覆盖数"打分，分数低的优先替换
    requested = set(requirements.keys())
    candidate_indices = list(range(len(sampled)))

    def _score(idx: int) -> tuple[int, int]:
        archs = sampled_archetypes[idx]
        useful = len(archs & requested)
        # 同分时优先换 normal（保留 elite/boss 等 oversample 来的）
        is_normal = _spec_encounter_type(sampled[idx]) == "normal"
        return (useful, 0 if is_normal else 1)

    candidate_indices.sort(key=_score)  # 升序：archetype 少 + normal 在前
    drop_indices = set(candidate_indices[:len(additions)])

    out: list[EncounterSpec] = []
    for i, s in enumerate(sampled):
        if i not in drop_indices:
            out.append(s)
    out.extend(additions)
    return out


def _parse_archetype_min_count(spec_str: str) -> dict[str, int]:
    """解析 ``"multi_hit=2,aoe=2,power_build=2"`` 格式为 dict。"""
    if not spec_str or not str(spec_str).strip():
        return {}
    out: dict[str, int] = {}
    for token in str(spec_str).split(","):
        token = token.strip()
        if "=" not in token:
            continue
        name, value = token.split("=", 1)
        name = name.strip().lower()
        try:
            count = int(value.strip())
        except ValueError:
            continue
        if name and count > 0:
            out[name] = count
    return out


def filter_encounter_pool(pool: list[EncounterSpec], query: str = "") -> list[EncounterSpec]:
    """Filter by comma-separated encounter id, tag, key, or label fragments."""
    raw = (query or "").strip()
    if not raw:
        return list(pool)
    tokens = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not tokens:
        return list(pool)

    selected: list[EncounterSpec] = []
    for spec in pool:
        haystack = " ".join(
            [
                spec.encounter_id,
                spec.tag,
                encounter_key(spec),
                encounter_label(spec),
            ]
        ).lower()
        if any(token in haystack for token in tokens):
            selected.append(spec)
    return selected


VALID_TIERS = ("normal", "elite", "boss")


def classify_tier(spec: EncounterSpec) -> str:
    """Map an encounter spec to one of {normal, elite, boss}.

    Source of truth is the tag suffix (e.g. ``skada_floor_08_elite``),
    which Skada datasets fill consistently. We additionally probe the
    encounter id (uppercase) so legacy / hand-written specs without a
    tag still classify correctly. Anything that doesn't match boss/elite
    falls back to normal — including atypical names like
    ``SEAPUNK_WEAK`` or ``OVERGROWTH_CRAWLERS`` which are act-1 normals.
    """
    tag = (spec.tag or "").lower()
    if "boss" in tag:
        return "boss"
    if "elite" in tag:
        return "elite"
    eid = (spec.encounter_id or "").upper()
    if "BOSS" in eid:
        return "boss"
    if "ELITE" in eid:
        return "elite"
    return "normal"


def filter_by_tier(pool: list[EncounterSpec], tiers: str = "") -> list[EncounterSpec]:
    """Keep only specs whose tier is in the comma-separated allow-list.

    Parameters
    ----------
    tiers:
        Comma-separated subset of ``{"normal", "elite", "boss"}``. Empty
        / ``"all"`` keeps everything (no filtering). Whitespace and casing
        are ignored.

    Notes
    -----
    Use this *before* stratified sampling / oversample ratios so the
    downstream sampler only sees relevant tiers. ``boss-oversample-ratio``
    + ``elite-oversample-ratio`` are still respected, just within the
    filtered pool.
    """
    raw = (tiers or "").strip().lower()
    if not raw or raw == "all":
        return list(pool)
    requested = {tok.strip() for tok in raw.split(",") if tok.strip()}
    invalid = requested - set(VALID_TIERS)
    if invalid:
        raise ValueError(
            f"unknown tier(s) {sorted(invalid)}; pick from {VALID_TIERS}"
        )
    if not requested:
        return list(pool)
    return [spec for spec in pool if classify_tier(spec) in requested]


__all__ = [
    "EncounterSpec",
    "VALID_TIERS",
    "build_fingerprint",
    "classify_tier",
    "encounter_key",
    "encounter_label",
    "filter_by_tier",
    "filter_encounter_pool",
    "load_skada_case_pool",
    "skada_case_to_spec",
]
