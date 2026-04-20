from __future__ import annotations

"""skada 人类对局 -> combat replay case.

V0 目标很克制：
- skada 暂时只负责提供“真实非战斗路径还原出的战斗开局 build”
- 不假装拥有逐步战斗动作标签
- 先把它变成 zero 可直接 rollout / evaluate 的固定 combat case

链路：
1. 从 `runs_full_detail` 里筛出版本/进阶/人数匹配的 run
2. 用 floor_timeline 把 starter build 回放成目标战斗开局 build
3. 生成 `SkadaCombatCase`
4. 用 `GameBridgeCombatRuntime(build=...)` 在 sim 中直接重开这场战斗
"""

import json
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..adapters.game_bridge import GameBridgeCombatRuntime
from ..buffers import ArtifactStore
from ..domain import (
    EvalSummary,
    FightLabel,
    RawTransition,
    TeacherLabel,
    TeacherRequest,
    assess_transition_progress,
    compute_fight_score,
    compute_hp_quality_score,
)


_SHARED_REPLAY_RUNTIMES: dict[tuple[int, bool, float], GameBridgeCombatRuntime] = {}


@dataclass(slots=True)
class SkadaBuild:
    deck: list[dict[str, int | str]] = field(default_factory=list)
    relics: list[dict[str, str]] = field(default_factory=list)
    current_hp: int = 0
    max_hp: int = 0
    max_energy: int = 3
    gold: int = 0

    def to_build_dict(self) -> dict[str, object]:
        return asdict(self)

    def clone(self) -> "SkadaBuild":
        return SkadaBuild(
            deck=[dict(card) for card in self.deck],
            relics=[dict(relic) for relic in self.relics],
            current_hp=self.current_hp,
            max_hp=self.max_hp,
            max_energy=self.max_energy,
            gold=self.gold,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SkadaBuild":
        return cls(
            deck=[dict(card) for card in payload.get("deck") or []],
            relics=[dict(relic) for relic in payload.get("relics") or []],
            current_hp=int(payload.get("current_hp") or 0),
            max_hp=int(payload.get("max_hp") or 0),
            max_energy=int(payload.get("max_energy") or 3),
            gold=int(payload.get("gold") or 0),
        )


@dataclass(slots=True)
class SkadaCombatCase:
    source_path: str
    source_line: int
    run_id: int
    seed: str
    game_version: str
    character_id: str
    ascension: int
    player_count: int
    floor: int
    encounter_id: str
    encounter_type: str
    won: bool
    build: SkadaBuild
    floor_state: dict[str, object] = field(default_factory=dict)
    card_usage: dict[str, dict[str, float | int]] = field(default_factory=dict)
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    @property
    def case_id(self) -> str:
        encounter_slug = self.encounter_id.lower()
        return f"run_{self.run_id}_floor_{self.floor}_{encounter_slug}"

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "source_path": self.source_path,
            "source_line": self.source_line,
            "run_id": self.run_id,
            "seed": self.seed,
            "game_version": self.game_version,
            "character_id": self.character_id,
            "ascension": self.ascension,
            "player_count": self.player_count,
            "floor": self.floor,
            "encounter_id": self.encounter_id,
            "encounter_type": self.encounter_type,
            "won": self.won,
            "build": self.build.to_build_dict(),
            "floor_state": self.floor_state,
            "card_usage": self.card_usage,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SkadaCombatCase":
        return cls(
            source_path=str(payload.get("source_path") or ""),
            source_line=int(payload.get("source_line") or 0),
            run_id=int(payload.get("run_id") or 0),
            seed=str(payload.get("seed") or ""),
            game_version=str(payload.get("game_version") or ""),
            character_id=str(payload.get("character_id") or ""),
            ascension=int(payload.get("ascension") or 0),
            player_count=int(payload.get("player_count") or 0),
            floor=int(payload.get("floor") or 0),
            encounter_id=str(payload.get("encounter_id") or ""),
            encounter_type=str(payload.get("encounter_type") or ""),
            won=bool(payload.get("won", False)),
            build=SkadaBuild.from_dict(_coerce_dict(payload.get("build"))),
            floor_state=dict(payload.get("floor_state") or {}),
            card_usage={str(key): dict(value) for key, value in (payload.get("card_usage") or {}).items()},
            metadata=dict(payload.get("metadata") or {}),
        )


def load_skada_run_record(path: Path, line_no: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if index == line_no:
                return json.loads(line)
    raise ValueError(f"line {line_no} not found in {path}")


def find_first_matching_run(
    *,
    root: Path,
    game_version: str,
    ascension: int,
    player_count: int = 1,
    character_id: str | None = None,
    victory_only: bool = False,
) -> tuple[Path, int]:
    candidate_roots = [root / "victory" / "details"]
    if not victory_only:
        candidate_roots.append(root / "failure" / "details")

    for details_root in candidate_roots:
        for path in sorted(details_root.glob("*.jsonl")):
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    obj = json.loads(line)
                    run = obj.get("run", {})
                    if run.get("game_version") != game_version:
                        continue
                    if int(run.get("ascension") or 0) != ascension:
                        continue
                    if int(run.get("player_count") or 0) != player_count:
                        continue
                    if character_id and str(run.get("character") or "").upper() != character_id.upper():
                        continue
                    if not (obj.get("combats") or []):
                        continue
                    return path, line_no
    raise LookupError(
        f"未找到 skada run: version={game_version} ascension={ascension} "
        f"player_count={player_count} character={character_id or '*'}"
    )


def iter_matching_run_records(
    *,
    root: Path,
    game_version: str,
    ascension: int,
    player_count: int = 1,
    character_id: str | None = None,
    victory_only: bool = False,
    max_runs: int | None = None,
) -> Iterator[tuple[Path, int, dict[str, Any]]]:
    emitted = 0
    candidate_roots = [root / "victory" / "details"]
    if not victory_only:
        candidate_roots.append(root / "failure" / "details")

    for details_root in candidate_roots:
        for path in sorted(details_root.glob("*.jsonl")):
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    obj = json.loads(line)
                    run = obj.get("run", {})
                    if run.get("game_version") != game_version:
                        continue
                    if int(run.get("ascension") or 0) != ascension:
                        continue
                    if int(run.get("player_count") or 0) != player_count:
                        continue
                    if character_id and str(run.get("character") or "").upper() != character_id.upper():
                        continue
                    if not (obj.get("combats") or []):
                        continue
                    yield path, line_no, obj
                    emitted += 1
                    if max_runs is not None and emitted >= max_runs:
                        return


def resolve_starting_build_from_runtime(
    *,
    character_id: str,
    encounter_id: str,
    port: int = 15527,
    auto_launch: bool = True,
    connect_timeout_s: float = 30.0,
    seed: str | None = None,
    build: dict[str, object] | None = None,
) -> SkadaBuild:
    runtime = GameBridgeCombatRuntime(
        port=port,
        auto_launch=auto_launch,
        connect_timeout_s=connect_timeout_s,
        character_id=character_id,
        encounter_id=encounter_id,
        seed=seed,
        build=build,
    )
    try:
        state = runtime.reset()
    finally:
        runtime.close()

    battle_raw = state.raw.get("battle")
    battle_player_raw = battle_raw.get("player") if isinstance(battle_raw, dict) else {}
    player_raw = _coerce_dict(state.raw.get("player")) or _coerce_dict(battle_player_raw)
    deck = [
        {
            "id": str(card.get("id") or ""),
            "upgrade_level": int(card.get("upgrades") or 0),
        }
        for card in (player_raw.get("deck") or [])
    ]
    relics = [{"id": str(relic.get("id") or "")} for relic in (player_raw.get("relics") or [])]
    starter_template = default_starter_build(character_id)
    if not deck:
        deck = [dict(card) for card in starter_template.deck]
    if not relics:
        relics = [dict(relic) for relic in starter_template.relics]
    return SkadaBuild(
        deck=deck,
        relics=relics,
        current_hp=int(player_raw.get("current_hp", player_raw.get("hp", 0)) or 0),
        max_hp=int(player_raw.get("max_hp", 0) or 0),
        max_energy=int(player_raw.get("max_energy", 3) or 3),
        gold=int(player_raw.get("gold", 0) or 0),
    )


def default_starter_build(character_id: str, *, db_path: Path | None = None) -> SkadaBuild:
    normalized = character_id.strip().upper()
    resolved_db_path = db_path or _default_game_wiki_db_path()
    return _load_starter_build_from_db(normalized, str(resolved_db_path))


def _coerce_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


_default_starter_template = default_starter_build


def _default_game_wiki_db_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "game_wiki" / "game_catalog.sqlite"


@lru_cache(maxsize=32)
def _load_starter_build_from_db(character_id: str, db_path: str) -> SkadaBuild:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(
            f"缺少权威 game_wiki 数据库: {path}。"
            " 请先运行 STS2AI/data/export_game_catalog_runtime.py 导出。"
        )
    con = sqlite3.connect(str(path))
    try:
        row = con.execute(
            "SELECT starting_deck_json, starting_relics_json, starting_potions_json, starting_hp, starting_gold, max_energy "
            "FROM characters WHERE UPPER(id)=?",
            (character_id.upper(),),
        ).fetchone()
    finally:
        con.close()
    if row is None:
        raise ValueError(f"game_wiki 中未找到角色 starter build: {character_id}")
    starting_deck_json, starting_relics_json, _, starting_hp, starting_gold, max_energy = row
    deck_ids = json.loads(starting_deck_json or "[]")
    relic_ids = json.loads(starting_relics_json or "[]")
    return SkadaBuild(
        deck=[{"id": str(card_id), "upgrade_level": 0} for card_id in deck_ids],
        relics=[{"id": str(relic_id)} for relic_id in relic_ids],
        current_hp=int(starting_hp or 0),
        max_hp=int(starting_hp or 0),
        max_energy=int(max_energy or 3),
        gold=int(starting_gold or 0),
    )


def build_case_from_record(
    record: dict[str, Any],
    *,
    source_path: Path,
    source_line: int,
    starter_build: SkadaBuild,
    combat_index: int = 0,
) -> SkadaCombatCase:
    run = record.get("run", {})
    combats = sorted(record.get("combats") or [], key=lambda item: int(item.get("floor") or 0))
    if not combats:
        raise ValueError("skada run does not contain combats")
    combat = combats[combat_index]
    floor = int(combat.get("floor") or 0)
    timeline = _timeline_by_floor(record.get("floor_timeline") or [])
    build = _reconstruct_build_before_floor(
        starter_build=starter_build,
        floor_timeline=record.get("floor_timeline") or [],
        combat_floor=floor,
        fallback_hp=int((timeline.get(floor) or {}).get("hp_before") or starter_build.current_hp or starter_build.max_hp or 0),
        fallback_gold=int((timeline.get(floor) or {}).get("gold_before") or starter_build.gold or 0),
    )
    return SkadaCombatCase(
        source_path=str(source_path),
        source_line=source_line,
        run_id=int(run.get("run_id") or 0),
        seed=str(run.get("seed") or ""),
        game_version=str(run.get("game_version") or ""),
        character_id=str(run.get("character") or ""),
        ascension=int(run.get("ascension") or 0),
        player_count=int(run.get("player_count") or 0),
        floor=floor,
        encounter_id=str(combat.get("encounter") or ""),
        encounter_type=str(combat.get("type") or combat.get("enc_type") or ""),
        won=bool(combat.get("won", False)),
        build=build,
        floor_state={
            "hp_before": int((timeline.get(floor) or {}).get("hp_before") or 0),
            "hp_after": int((timeline.get(floor) or {}).get("hp_after") or 0),
            "gold_before": int((timeline.get(floor) or {}).get("gold_before") or 0),
            "gold_after": int((timeline.get(floor) or {}).get("gold_after") or 0),
        },
        card_usage=_build_card_usage_map(combat),
        metadata={
            "combat_turns": int(combat.get("turns") or 0),
            "run_floor_reached": int(run.get("floor_reached") or 0),
            "run_victory": bool(run.get("is_victory", False)),
        },
    )


def build_cases_from_record(
    record: dict[str, Any],
    *,
    source_path: Path,
    source_line: int,
    starter_build: SkadaBuild,
    max_combats: int | None = None,
) -> list[SkadaCombatCase]:
    combats = sorted(record.get("combats") or [], key=lambda item: int(item.get("floor") or 0))
    if max_combats is not None:
        combats = combats[:max_combats]
    cases = []
    for combat_index, _ in enumerate(combats):
        cases.append(
            build_case_from_record(
                record,
                source_path=source_path,
                source_line=source_line,
                starter_build=starter_build,
                combat_index=combat_index,
            )
        )
    return cases


def load_case_index(path: Path) -> list[SkadaCombatCase]:
    cases: list[SkadaCombatCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            cases.append(SkadaCombatCase.from_dict(json.loads(text)))
    return cases


def _timeline_by_floor(rows: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    mapping: dict[int, dict[str, Any]] = {}
    for row in rows:
        floor = int(row.get("floor") or 0)
        if floor > 0:
            mapping[floor] = row
    return mapping


def _reconstruct_build_before_floor(
    *,
    starter_build: SkadaBuild,
    floor_timeline: list[dict[str, Any]],
    combat_floor: int,
    fallback_hp: int,
    fallback_gold: int,
) -> SkadaBuild:
    build = starter_build.clone()
    for row in sorted(floor_timeline, key=lambda item: int(item.get("floor") or 0)):
        floor = int(row.get("floor") or 0)
        if floor <= 0 or floor >= combat_floor:
            continue
        _apply_relic_choices(build, row.get("relic_choices") or [])
        _apply_card_choices(build, row.get("card_choices") or [])
        _apply_card_upgrades(build, row.get("card_upgrades") or [])
        _apply_shop_actions(build, row.get("shop_actions") or [])

    build.current_hp = max(0, int(fallback_hp))
    build.gold = max(0, int(fallback_gold))
    build.max_hp = max(int(build.max_hp), int(build.current_hp), 1)
    return build


def _apply_relic_choices(build: SkadaBuild, relic_choices: list[dict[str, Any]]) -> None:
    owned = {str(relic.get("id") or "").upper() for relic in build.relics}
    for choice in relic_choices:
        if not bool(choice.get("was_picked", False)):
            continue
        relic_id = str(choice.get("relic_id") or "").upper()
        if relic_id and relic_id not in owned:
            build.relics.append({"id": relic_id})
            owned.add(relic_id)


def _apply_card_choices(build: SkadaBuild, card_choices: list[dict[str, Any]]) -> None:
    for choice in card_choices:
        if not bool(choice.get("was_picked", False)):
            continue
        card_id, upgrade_level = _split_upgrade_suffix(str(choice.get("card_id") or ""))
        if card_id:
            build.deck.append({"id": card_id, "upgrade_level": upgrade_level})


def _apply_card_upgrades(build: SkadaBuild, upgrades: list[dict[str, Any]]) -> None:
    for entry in upgrades:
        target_id, target_level = _split_upgrade_suffix(str(entry.get("card_id") or ""))
        if not target_id:
            continue
        candidate_indices = [
            index
            for index, card in enumerate(build.deck)
            if str(card.get("id") or "").upper() == target_id
        ]
        if not candidate_indices:
            build.deck.append({"id": target_id, "upgrade_level": max(1, target_level)})
            continue
        preferred_index = next(
            (
                index
                for index in candidate_indices
                if int(build.deck[index].get("upgrade_level") or 0) < max(1, target_level)
            ),
            candidate_indices[0],
        )
        build.deck[preferred_index]["upgrade_level"] = max(int(build.deck[preferred_index].get("upgrade_level") or 0), max(1, target_level))


def _apply_shop_actions(build: SkadaBuild, shop_actions: list[dict[str, Any]]) -> None:
    owned_relics = {str(relic.get("id") or "").upper() for relic in build.relics}
    for action in shop_actions:
        action_type = str(action.get("action_type") or "").lower()
        item_id, upgrade_level = _split_upgrade_suffix(str(action.get("item_id") or ""))
        if action_type == "remove":
            for index, card in enumerate(build.deck):
                if str(card.get("id") or "").upper() == item_id:
                    del build.deck[index]
                    break
        elif action_type == "buy_card" and item_id:
            build.deck.append({"id": item_id, "upgrade_level": upgrade_level})
        elif action_type == "buy_relic" and item_id and item_id not in owned_relics:
            build.relics.append({"id": item_id})
            owned_relics.add(item_id)


def _split_upgrade_suffix(card_id: str) -> tuple[str, int]:
    normalized = card_id.strip().upper()
    if not normalized:
        return "", 0
    upgrade_level = len(normalized) - len(normalized.rstrip("+"))
    base_id = normalized.rstrip("+")
    return base_id, upgrade_level


def _build_card_usage_map(combat: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    usage: dict[str, dict[str, float | int]] = {}
    for row in combat.get("card_combat_perf") or []:
        card_id = str(row.get("card_id") or "").upper()
        if not card_id:
            continue
        usage[card_id] = {
            "plays": int(row.get("plays") or 0),
            "damage": float(row.get("damage") or 0.0),
            "block": float(row.get("block") or 0.0),
            "energy": float(row.get("energy") or 0.0),
        }
    return usage


class AggregateCardUsageTeacher:
    """Weak teacher from human combat aggregates.

    This is intentionally not a true oracle. It only says:
    - actions whose card ids were used more / dealt more / blocked more in the
      human fight should receive higher prior weight
    - unmatched actions (including end_turn) get a small fallback score
    """

    def __init__(self, case: SkadaCombatCase):
        self._case = case
        self._score_table = {
            card_id: (
                1.0
                + float(stats.get("plays") or 0.0)
                + 0.05 * float(stats.get("damage") or 0.0)
                + 0.05 * float(stats.get("block") or 0.0)
            )
            for card_id, stats in case.card_usage.items()
        }

    def label_request(self, request: TeacherRequest, runtime_factory=None, seed: str | None = None) -> TeacherLabel:
        sample = request.sample
        if not sample.legal_actions:
            return TeacherLabel(teacher_value=float(self._case.won))
        scores = []
        for action in sample.legal_actions:
            score = self._score_table.get(action.card_id.upper(), 0.05)
            if action.action_type == "end_turn":
                score = 0.01
            scores.append(float(score))
        total = sum(scores) or 1.0
        policy = [score / total for score in scores]
        best_action_index = max(range(len(scores)), key=lambda idx: scores[idx])
        ordered = sorted(scores, reverse=True)
        margin = float(ordered[0] - ordered[1]) if len(ordered) >= 2 else float(ordered[0])
        return TeacherLabel(
            policy=policy,
            topk_indices=sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[: min(3, len(scores))],
            best_action_index=best_action_index,
            ranking_margin=margin,
            teacher_value=FightLabel(
                fight_win=1.0 if self._case.won else 0.0,
                enemy_hp_fraction_dealt=1.0 if self._case.won else 0.7,
                self_hp_fraction_remaining=max(
                    0.0,
                    min(
                        1.0,
                        float(self._case.floor_state.get("hp_after", 0) or 0)
                        / float(max(int(self._case.build.max_hp), 1)),
                    ),
                ),
                player_hp=float(self._case.floor_state.get("hp_after", 0) or 0),
                player_max_hp=float(max(int(self._case.build.max_hp), 1)),
            ).fight_score,
            metadata={"teacher": "AggregateCardUsageTeacher"},
        )


class MultiCaseAggregateTeacher:
    """Dispatch aggregate card-usage teachers by replay case id.

    This keeps the V0 teacher simple while allowing one training job to mix
    many skada-derived combat roots.
    """

    def __init__(self, cases: Iterable[SkadaCombatCase]):
        teacher_map = {case.case_id: AggregateCardUsageTeacher(case) for case in cases}
        if not teacher_map:
            raise ValueError("MultiCaseAggregateTeacher 需要至少一个 case。")
        self._teachers = teacher_map

    def label_request(self, request: TeacherRequest, runtime_factory=None, seed: str | None = None) -> TeacherLabel:
        case_id = str(request.sample.state.context.metadata.get("skada_case_id") or "")
        teacher = self._teachers.get(case_id)
        if teacher is None:
            fallback_teacher = next(iter(self._teachers.values()))
            label = fallback_teacher.label_request(request, runtime_factory=runtime_factory, seed=seed)
            metadata = dict(label.metadata)
            metadata["teacher_fallback_case_id"] = case_id
            label.metadata = metadata
            return label
        return teacher.label_request(request, runtime_factory=runtime_factory, seed=seed)


class SkadaReplayRuntime:
    """BattleRuntime wrapper that annotates states with stable skada case metadata."""

    def __init__(
        self,
        case: SkadaCombatCase,
        *,
        port: int = 15527,
        auto_launch: bool = True,
        connect_timeout_s: float = 30.0,
    ):
        self._case = case
        runtime_key = (int(port), bool(auto_launch), float(connect_timeout_s))
        runtime = _SHARED_REPLAY_RUNTIMES.get(runtime_key)
        if runtime is None:
            runtime = GameBridgeCombatRuntime(
                port=port,
                auto_launch=auto_launch,
                connect_timeout_s=connect_timeout_s,
                character_id=case.character_id,
                encounter_id=case.encounter_id,
                seed=case.seed,
                build=case.build.to_build_dict(),
            )
            _SHARED_REPLAY_RUNTIMES[runtime_key] = runtime
        self._runtime = runtime
        self._runtime.configure(
            character_id=case.character_id,
            encounter_id=case.encounter_id,
            seed=case.seed,
            build=case.build.to_build_dict(),
        )

    def reset(self, *, seed: str | None = None):
        self._runtime.configure(
            character_id=self._case.character_id,
            encounter_id=self._case.encounter_id,
            seed=seed or self._case.seed,
            build=self._case.build.to_build_dict(),
        )
        return self._decorate_state(self._runtime.reset(seed=seed))

    def get_state(self):
        return self._decorate_state(self._runtime.get_state())

    def step(self, action_index: int):
        return self._decorate_state(self._runtime.step(action_index))

    def close(self) -> None:
        # 训练/评估会频繁创建轻量 runtime wrapper，这里保留底层 session，
        # 让后续 episode 继续复用 reset-only 路径，真正关闭交给进程级清理。
        return None

    def _decorate_state(self, state):
        metadata = dict(state.context.metadata)
        metadata.update(
            {
                "skada_case_id": self._case.case_id,
                "skada_run_id": self._case.run_id,
                "skada_floor": self._case.floor,
                "skada_source_line": self._case.source_line,
                "skada_source_path": self._case.source_path,
            }
        )
        state.context.metadata = metadata
        return state


def close_shared_replay_runtimes() -> None:
    for runtime in list(_SHARED_REPLAY_RUNTIMES.values()):
        runtime.close()
    _SHARED_REPLAY_RUNTIMES.clear()


class OrderedRunRuntimeFactory:
    """按同一条 run 的战斗顺序推进。

    规则：
    - 当前战斗胜利：进入下一场
    - 当前战斗失败或 timeout：重置回第一场
    - 打完最后一场并胜利：也重置回第一场，开始下一次 run attempt
    """

    def __init__(
        self,
        cases: list[SkadaCombatCase],
        *,
        port: int = 15527,
        auto_launch: bool = True,
        connect_timeout_s: float = 30.0,
    ):
        if not cases:
            raise ValueError("OrderedRunRuntimeFactory 需要至少一个 case。")
        self._cases = list(cases)
        self._port = port
        self._auto_launch = auto_launch
        self._connect_timeout_s = connect_timeout_s
        self._index = 0

    def __call__(self) -> SkadaReplayRuntime:
        case = self._cases[self._index]
        return SkadaReplayRuntime(
            case,
            port=self._port,
            auto_launch=self._auto_launch,
            connect_timeout_s=self._connect_timeout_s,
        )

    def on_episode_end(self, event: dict[str, object]) -> None:
        outcome = str(event.get("outcome") or "").strip().lower()
        truncated = bool(event.get("truncated", False))
        success = outcome in {"victory", "win"} and not truncated
        if success and self._index < len(self._cases) - 1:
            self._index += 1
            return
        self._index = 0

    def clone_for_port(self, port: int) -> "OrderedRunRuntimeFactory":
        return OrderedRunRuntimeFactory(
            list(self._cases),
            port=port,
            auto_launch=self._auto_launch,
            connect_timeout_s=self._connect_timeout_s,
        )

    @property
    def current_case_id(self) -> str:
        return self._cases[self._index].case_id


class FixedSkadaCaseEvaluator:
    """Evaluate a policy on one or more fixed skada-derived combat roots.

    固定评估模式下，每个 case 都独立重复 `episodes_per_case` 次；
    不共享前一场战斗的成败。
    """

    def __init__(
        self,
        cases: list[SkadaCombatCase],
        *,
        port: int = 15527,
        auto_launch: bool = True,
        connect_timeout_s: float = 30.0,
        episodes_per_case: int = 1,
        artifact_store: ArtifactStore | None = None,
    ):
        self._cases = cases
        self._port = port
        self._auto_launch = auto_launch
        self._connect_timeout_s = connect_timeout_s
        self._episodes_per_case = max(1, int(episodes_per_case))
        self._artifact_store = artifact_store
        self._trace_name = "eval_trace"

    def set_trace_context(self, *, iteration: int, phase: str) -> None:
        self._trace_name = f"iter_{iteration:04d}_{phase}"

    def evaluate(self, policy) -> list[EvalSummary]:
        summaries: list[EvalSummary] = []
        for case_index, case in enumerate(self._cases):
            episode_labels: list[FightLabel] = []
            episode_metrics: list[dict[str, float | int | bool]] = []
            agreement_hits = 0.0
            overlap_hits = 0.0
            agreement_steps = 0
            teacher = AggregateCardUsageTeacher(case)
            for episode_index in range(self._episodes_per_case):
                result = _rollout_case_episode(
                    case=case,
                    policy=policy,
                    port=self._port,
                    auto_launch=self._auto_launch,
                    connect_timeout_s=self._connect_timeout_s,
                    trace_name=self._trace_name,
                    artifact_store=self._artifact_store,
                    teacher=teacher,
                    case_index=case_index,
                    episode_index=episode_index,
                )
                episode_labels.append(result["label"])
                episode_metrics.append(result["metrics"])
                agreement_hits += float(result["agreement_hits"])
                overlap_hits += float(result["overlap_hits"])
                agreement_steps += int(result["agreement_steps"])

            summaries.append(
                _build_case_eval_summary(
                    case=case,
                    labels=episode_labels,
                    metrics=episode_metrics,
                    agreement_hits=agreement_hits,
                    overlap_hits=overlap_hits,
                    agreement_steps=agreement_steps,
                    metadata_extra={},
                )
            )
        return summaries

    def _append_eval_trace_row(self, row: dict[str, object]) -> None:
        if self._artifact_store is None:
            return
        self._artifact_store.append_eval_trace_row(self._trace_name, row)


class OrderedRunCaseEvaluator:
    """按整条 run attempt 评估 policy。

    每次 attempt 都从第一场 combat 开始，遇到失败或 timeout 就停止，不再评估后续 combat。
    最后仍然返回固定 cohort 列表；未到达的 case `num_episodes=0`。
    """

    def __init__(
        self,
        cases: list[SkadaCombatCase],
        *,
        port: int = 15527,
        auto_launch: bool = True,
        connect_timeout_s: float = 30.0,
        episodes_per_case: int = 1,
        artifact_store: ArtifactStore | None = None,
    ):
        if not cases:
            raise ValueError("OrderedRunCaseEvaluator 需要至少一个 case。")
        self._cases = list(cases)
        self._port = port
        self._auto_launch = auto_launch
        self._connect_timeout_s = connect_timeout_s
        self._run_attempts = max(1, int(episodes_per_case))
        self._artifact_store = artifact_store
        self._trace_name = "eval_trace"

    def set_trace_context(self, *, iteration: int, phase: str) -> None:
        self._trace_name = f"iter_{iteration:04d}_{phase}"

    def evaluate(self, policy) -> list[EvalSummary]:
        per_case_labels: dict[str, list[FightLabel]] = defaultdict(list)
        per_case_metrics: dict[str, list[dict[str, float | int | bool]]] = defaultdict(list)
        per_case_agreement: dict[str, list[tuple[float, float, int]]] = defaultdict(list)

        for attempt_index in range(self._run_attempts):
            for case_index, case in enumerate(self._cases):
                result = _rollout_case_episode(
                    case=case,
                    policy=policy,
                    port=self._port,
                    auto_launch=self._auto_launch,
                    connect_timeout_s=self._connect_timeout_s,
                    trace_name=self._trace_name,
                    artifact_store=self._artifact_store,
                    teacher=AggregateCardUsageTeacher(case),
                    case_index=case_index,
                    episode_index=attempt_index,
                )
                per_case_labels[case.case_id].append(result["label"])
                per_case_metrics[case.case_id].append(result["metrics"])
                per_case_agreement[case.case_id].append(
                    (
                        float(result["agreement_hits"]),
                        float(result["overlap_hits"]),
                        int(result["agreement_steps"]),
                    )
                )
                if not bool(result["success"]):
                    break

        summaries: list[EvalSummary] = []
        for case in self._cases:
            labels = per_case_labels.get(case.case_id, [])
            metrics = per_case_metrics.get(case.case_id, [])
            agreement_stats = per_case_agreement.get(case.case_id, [])
            agreement_hits = sum(item[0] for item in agreement_stats)
            overlap_hits = sum(item[1] for item in agreement_stats)
            agreement_steps = sum(item[2] for item in agreement_stats)
            summaries.append(
                _build_case_eval_summary(
                    case=case,
                    labels=labels,
                    metrics=metrics,
                    agreement_hits=agreement_hits,
                    overlap_hits=overlap_hits,
                    agreement_steps=agreement_steps,
                    metadata_extra={
                        "run_attempts": self._run_attempts,
                        "reached_episodes": len(labels),
                        "cohort_mode": "in_domain" if len({item.run_id for item in self._cases}) == 1 else "generalization",
                    },
                )
            )
        return summaries


def _rollout_case_episode(
    *,
    case: SkadaCombatCase,
    policy,
    port: int,
    auto_launch: bool,
    connect_timeout_s: float,
    trace_name: str,
    artifact_store: ArtifactStore | None,
    teacher: AggregateCardUsageTeacher,
    case_index: int,
    episode_index: int,
) -> dict[str, object]:
    """Roll out one replay case and emit the same trace schema for all evaluators.

    这里显式记录 reset / policy / env_step / trace 写盘耗时，方便区分：
    - 是模型推理慢
    - 还是 simulator step 慢
    - 还是大量超时 fight 把总体 wall time 拉长
    """
    fight_started_at = time.perf_counter()
    reset_duration_s = 0.0
    policy_select_duration_s = 0.0
    env_step_duration_s = 0.0
    observe_duration_s = 0.0
    trace_write_duration_s = 0.0
    runtime = SkadaReplayRuntime(
        case,
        port=port,
        auto_launch=auto_launch,
        connect_timeout_s=connect_timeout_s,
    )
    try:
        reset_hook = getattr(policy, "reset_episode", None)
        if callable(reset_hook):
            reset_hook()
        reset_started_at = time.perf_counter()
        state = runtime.reset()
        reset_duration_s = time.perf_counter() - reset_started_at
        step_count = 0
        progress_steps = 0
        no_progress_steps = 0
        max_no_progress_streak = 0
        current_no_progress_streak = 0
        agreement_hits = 0.0
        overlap_hits = 0.0
        agreement_steps = 0
        for _ in range(200):
            if state.terminal or not state.legal_actions:
                break
            select_started_at = time.perf_counter()
            action_index = policy.select_action(state)
            policy_select_duration_s += time.perf_counter() - select_started_at
            teacher_label = _teacher_label_for_actions(teacher, state.legal_actions)
            if teacher_label.best_action_index >= 0:
                agreement_steps += 1
                agreement_hits += 1.0 if action_index == teacher_label.best_action_index else 0.0
                overlap_hits += 1.0 if action_index in teacher_label.topk_indices else 0.0
            chosen_action = state.legal_actions[action_index] if state.legal_actions else None
            env_step_started_at = time.perf_counter()
            next_state = runtime.step(action_index)
            env_step_duration_s += time.perf_counter() - env_step_started_at
            progress = assess_transition_progress(state, next_state)
            if progress.made_progress:
                progress_steps += 1
                current_no_progress_streak = 0
            else:
                no_progress_steps += 1
                current_no_progress_streak += 1
                max_no_progress_streak = max(max_no_progress_streak, current_no_progress_streak)
            serialized_transition = RawTransition(
                run_id=str(case.run_id),
                fight_id=f"{case.case_id}|ep{episode_index}",
                step_idx=step_count,
                seed=case.seed,
                action_index=action_index,
                state=state,
                action=chosen_action if chosen_action is not None else state.legal_actions[0],
                next_state=next_state,
                done=next_state.terminal,
                fight_outcome=next_state.run_outcome,
                run_outcome=next_state.run_outcome,
                metadata={
                    "uncertainty": float(getattr(policy, "estimate_uncertainty", lambda _state: 0.0)(state) or 0.0),
                    "top2_gap": 0.0,
                    "made_progress": bool(progress.made_progress),
                    "enemy_hp_delta": float(progress.enemy_hp_delta),
                    "enemy_count_delta": int(progress.enemy_count_delta),
                },
            ).to_dict()
            trace_started_at = time.perf_counter()
            _append_eval_trace_row(
                artifact_store,
                trace_name,
                {
                    "event": "step",
                    "case_index": case_index,
                    "episode_index": episode_index,
                    "case_id": case.case_id,
                    "run_id": case.run_id,
                    "floor": case.floor,
                    "encounter_id": case.encounter_id,
                    "step_idx": step_count,
                    "player_hp": state.player.hp,
                    "player_block": state.player.block,
                    "player_energy": state.player.energy,
                    "enemy_hp": [enemy.hp for enemy in state.enemies],
                    "enemy_block": [enemy.block for enemy in state.enemies],
                    "action_index": action_index,
                    "action_id": chosen_action.action_id if chosen_action is not None else "",
                    "card_id": chosen_action.card_id if chosen_action is not None else "",
                    "target_id": chosen_action.target_id if chosen_action is not None else "",
                    "state": serialized_transition["state"],
                    "action": serialized_transition["action"],
                    "next_state": serialized_transition["next_state"],
                    "teacher_best_action_index": teacher_label.best_action_index,
                    "teacher_topk_indices": teacher_label.topk_indices,
                    "made_progress": bool(progress.made_progress),
                    "enemy_hp_delta": float(progress.enemy_hp_delta),
                    "enemy_count_delta": int(progress.enemy_count_delta),
                },
            )
            trace_write_duration_s += time.perf_counter() - trace_started_at
            observe_hook = getattr(policy, "observe_transition", None)
            if callable(observe_hook):
                observe_started_at = time.perf_counter()
                observe_hook(state, action_index, next_state)
                observe_duration_s += time.perf_counter() - observe_started_at
            state = next_state
            step_count += 1
            if state.terminal:
                break
        truncated = bool(not state.terminal and step_count >= 200)
        duration_s = time.perf_counter() - fight_started_at
        accounted_duration_s = (
            reset_duration_s
            + policy_select_duration_s
            + env_step_duration_s
            + observe_duration_s
            + trace_write_duration_s
        )
        overhead_duration_s = max(0.0, duration_s - accounted_duration_s)
        metrics = {
            "truncated": truncated,
            "progress_steps": progress_steps,
            "no_progress_steps": no_progress_steps,
            "no_progress_ratio": (no_progress_steps / max(step_count, 1)),
            "max_no_progress_streak": max_no_progress_streak,
            "duration_s": duration_s,
            "step_throughput": step_count / max(duration_s, 1e-6),
            "core_step_throughput": step_count / max(policy_select_duration_s + env_step_duration_s, 1e-6),
            "reset_duration_s": reset_duration_s,
            "policy_select_duration_s": policy_select_duration_s,
            "env_step_duration_s": env_step_duration_s,
            "observe_duration_s": observe_duration_s,
            "trace_write_duration_s": trace_write_duration_s,
            "overhead_duration_s": overhead_duration_s,
        }
        trace_started_at = time.perf_counter()
        _append_eval_trace_row(
            artifact_store,
            trace_name,
            {
                "event": "fight_end",
                "case_index": case_index,
                "episode_index": episode_index,
                "case_id": case.case_id,
                "run_id": case.run_id,
                "floor": case.floor,
                "encounter_id": case.encounter_id,
                "duration_s": round(duration_s, 6),
                "steps": step_count,
                "step_throughput": round(step_count / max(duration_s, 1e-6), 6),
                "core_step_throughput": round(
                    step_count / max(policy_select_duration_s + env_step_duration_s, 1e-6),
                    6,
                ),
                "reset_duration_s": round(reset_duration_s, 6),
                "policy_select_duration_s": round(policy_select_duration_s, 6),
                "env_step_duration_s": round(env_step_duration_s, 6),
                "observe_duration_s": round(observe_duration_s, 6),
                "trace_write_duration_s": round(trace_write_duration_s, 6),
                "overhead_duration_s": round(overhead_duration_s, 6),
                "outcome": "timeout" if truncated else str(state.run_outcome),
                "terminal": bool(state.terminal),
                "truncated": truncated,
                "final_player_hp": state.player.hp,
                "final_enemy_hp": [enemy.hp for enemy in state.enemies],
                "progress_steps": progress_steps,
                "no_progress_steps": no_progress_steps,
                "no_progress_ratio": round(no_progress_steps / max(step_count, 1), 6),
                "max_no_progress_streak": max_no_progress_streak,
            },
        )
        trace_write_duration_s += time.perf_counter() - trace_started_at
        success = bool(not truncated and str(state.run_outcome).lower() in {"victory", "win"})
        return {
            "label": _build_eval_label(state, truncated=truncated),
            "metrics": metrics,
            "agreement_hits": agreement_hits,
            "overlap_hits": overlap_hits,
            "agreement_steps": agreement_steps,
            "success": success,
        }
    finally:
        runtime.close()


def _build_case_eval_summary(
    *,
    case: SkadaCombatCase,
    labels: list[FightLabel],
    metrics: list[dict[str, float | int | bool]],
    agreement_hits: float,
    overlap_hits: float,
    agreement_steps: int,
    metadata_extra: dict[str, object],
) -> EvalSummary:
    aggregate = _aggregate_eval_labels(labels)
    timeout_count = sum(1 for item in metrics if bool(item.get("truncated", False)))
    avg_step_count = (
        sum(float(item.get("step_count", 0.0)) for item in metrics) / max(len(metrics), 1)
        if metrics
        else 0.0
    )
    avg_no_progress_ratio = (
        sum(float(item.get("no_progress_ratio", 0.0)) for item in metrics) / max(len(metrics), 1)
        if metrics
        else 0.0
    )
    avg_max_no_progress_streak = (
        sum(float(item.get("max_no_progress_streak", 0.0)) for item in metrics) / max(len(metrics), 1)
        if metrics
        else 0.0
    )
    fight_quality_score = compute_fight_score(
        aggregate,
        encounter_class=case.encounter_type,
        truncated=bool(timeout_count),
        no_progress_ratio=avg_no_progress_ratio,
        max_no_progress_streak=int(round(avg_max_no_progress_streak)),
        step_count=int(round(avg_step_count)),
    )
    hp_quality_score = compute_hp_quality_score(
        aggregate,
        encounter_class=case.encounter_type,
    )
    metadata = {
        "run_id": case.run_id,
        "floor": case.floor,
        "encounter_id": case.encounter_id,
        "encounter_type": case.encounter_type,
        "eval_bucket": str(case.encounter_type or "default").lower(),
        "num_episodes": len(labels),
        "avg_step_count": avg_step_count,
        "timeout_rate": timeout_count / max(len(metrics), 1) if metrics else 0.0,
        "avg_no_progress_ratio": avg_no_progress_ratio,
        "avg_max_no_progress_streak": avg_max_no_progress_streak,
        "fight_quality_score": fight_quality_score,
        "hp_quality_score": hp_quality_score,
        **metadata_extra,
    }
    return EvalSummary(
        cohort_name=f"skada_floor_{case.floor}_{case.encounter_id.lower()}",
        fight_win_rate=aggregate.fight_win,
        enemy_hp_fraction_dealt=aggregate.enemy_hp_fraction_dealt,
        self_hp_fraction_remaining=aggregate.self_hp_fraction_remaining,
        teacher_agreement_at_1=(agreement_hits / agreement_steps) if agreement_steps else 0.0,
        teacher_topk_overlap=(overlap_hits / agreement_steps) if agreement_steps else 0.0,
        metadata=metadata,
    )


def _append_eval_trace_row(
    artifact_store: ArtifactStore | None,
    trace_name: str,
    row: dict[str, object],
) -> None:
    if artifact_store is None:
        return
    artifact_store.append_eval_trace_row(trace_name, row)


def _build_eval_label(state, *, truncated: bool = False) -> FightLabel:
    enemy_max_hp = sum(enemy.max_hp for enemy in state.enemies) or 1.0
    enemy_remaining = sum(max(0.0, enemy.hp) for enemy in state.enemies)
    enemy_hp_fraction_dealt = max(0.0, min(1.0, 1.0 - enemy_remaining / enemy_max_hp))
    self_hp_fraction_remaining = 0.0
    if state.player.max_hp > 0:
        self_hp_fraction_remaining = max(0.0, min(1.0, state.player.hp / state.player.max_hp))
    if truncated:
        self_hp_fraction_remaining = 0.0
    fight_win = 1.0 if (not truncated and str(state.run_outcome).lower() in {"victory", "win"}) else 0.0
    return FightLabel(
        fight_win=fight_win,
        enemy_hp_fraction_dealt=enemy_hp_fraction_dealt,
        self_hp_fraction_remaining=self_hp_fraction_remaining,
        player_hp=max(0.0, float(state.player.hp)),
        player_max_hp=max(0.0, float(state.player.max_hp)),
    )


def _aggregate_eval_labels(labels: list[FightLabel]) -> FightLabel:
    if not labels:
        return FightLabel(fight_win=0.0, enemy_hp_fraction_dealt=0.0, self_hp_fraction_remaining=0.0)
    denom = float(len(labels))
    return FightLabel(
        fight_win=sum(label.fight_win for label in labels) / denom,
        enemy_hp_fraction_dealt=sum(label.enemy_hp_fraction_dealt for label in labels) / denom,
        self_hp_fraction_remaining=sum(label.self_hp_fraction_remaining for label in labels) / denom,
        player_hp=sum(label.player_hp for label in labels) / denom,
        player_max_hp=sum(label.player_max_hp for label in labels) / denom,
    )


def _teacher_label_for_actions(teacher: AggregateCardUsageTeacher, legal_actions) -> TeacherLabel:
    if not legal_actions:
        return TeacherLabel(best_action_index=-1)
    scores = []
    for action in legal_actions:
        score = teacher._score_table.get(action.card_id.upper(), 0.05)
        if action.action_type == "end_turn":
            score = 0.01
        scores.append(float(score))
    total = sum(scores) or 1.0
    policy = [score / total for score in scores]
    best_action_index = max(range(len(scores)), key=lambda idx: scores[idx])
    return TeacherLabel(
        policy=policy,
        topk_indices=sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[: min(3, len(scores))],
        best_action_index=best_action_index,
        ranking_margin=0.0,
        teacher_value=0.0,
    )
