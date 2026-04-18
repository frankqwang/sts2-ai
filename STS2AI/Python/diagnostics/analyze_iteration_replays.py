"""迭代回放分析：对比不同训练迭代的游戏回放。"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any
import sqlite3


FILENAME_RE = re.compile(
    r"^i(?P<iteration>\d+)_e(?P<episode>\d+)_(?P<tag>[A-Z]+)_f(?P<floor>\d+)\.txt$"
)
HEADER_RE = re.compile(
    r"^# outcome=(?P<outcome>\w+) floor=(?P<floor>\d+) combats=(?P<combats>\d+) "
    r"time=(?P<time>[0-9.]+)s end_reason=(?P<end_reason>\w+) error=(?P<error>.+)$"
)
COMBAT_START_RE = re.compile(
    r"^\[(?P<step>\d+)\] COMBAT #(?P<combat_index>\d+) floor=(?P<floor>\d+) "
    r"hp=(?P<hp>\d+) blk=(?P<blk>\d+) e=(?P<energy>\d+) hand=\[(?P<hand>.*)\] "
    r"intent=\[(?P<intent>.*)\]$"
)
COMBAT_NN_RE = re.compile(r"^\[(?P<step>\d+)\] COMBAT nn: (?P<body>.+)$")
TERMINAL_RE = re.compile(
    r"^\[(?P<step>\d+)\] TERMINAL: death floor=(?P<floor>\d+) hp=(?P<hp>\d+)/(?P<max_hp>\d+) "
    r"death_by=(?P<death_by>.+)$"
)
MAP_RE = re.compile(
    r"^\[(?P<step>\d+)\] map: (?P<label>.+?) \(idx=.* floor=(?P<floor>\d+) hp=(?P<hp>\d+)"
)
CARD_REWARD_RE = re.compile(
    r"^\[(?P<step>\d+)\] card_reward: (?P<label>.+?) \(idx=.* floor=(?P<floor>\d+) hp=(?P<hp>\d+)"
)
SHOP_RE = re.compile(
    r"^\[(?P<step>\d+)\] shop: (?P<label>.+?) \(idx=.* floor=(?P<floor>\d+) hp=(?P<hp>\d+)"
)
REST_RE = re.compile(
    r"^\[(?P<step>\d+)\] rest_site: (?P<label>.+?) \(idx=.* floor=(?P<floor>\d+) hp=(?P<hp>\d+)"
)
EVENT_RE = re.compile(
    r"^\[(?P<step>\d+)\] event(?: \[(?P<event_id>[^\]]+)\])?: (?P<label>.+?) "
    r"\(idx=.* floor=(?P<floor>\d+) hp=(?P<hp>\d+)"
)
REPEAT_RE = re.compile(r"REPEAT x(?P<count>\d+)")
ENEMY_RE = re.compile(r"([A-Z0-9_]+)\[([^\]]*)\]")
TOKEN_RE = re.compile(r"^[A-Z0-9_]+$")


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _pct(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _top(counter: Counter[str], limit: int = 10) -> list[dict[str, Any]]:
    return [
        {"name": name, "count": count}
        for name, count in counter.most_common(limit)
    ]


def _load_metrics_entry(metrics_path: Path, iteration: int) -> dict[str, Any] | None:
    if not metrics_path.exists():
        return None
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            entry = json.loads(text)
        except Exception:
            continue
        if int(entry.get("iteration", -1)) == iteration:
            return entry
    return None


def _enemy_tokens(intent_summary: str) -> list[str]:
    return [match.group(1) for match in ENEMY_RE.finditer(intent_summary or "")]


def _enemy_group(intent_summary: str) -> str:
    names = _enemy_tokens(intent_summary)
    return "+".join(names) if names else "UNKNOWN"


def _pretty_token(token: str) -> str:
    text = str(token or "").strip()
    if not text:
        return "未知"
    if TOKEN_RE.match(text):
        return " ".join(part.capitalize() for part in text.split("_"))
    return text


class SkadaNameResolver:
    def __init__(self) -> None:
        self.card_zh: dict[str, str] = {}
        self.encounter_zh: dict[str, str] = {}
        self.relic_zh: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        db_path = Path(__file__).resolve().parents[2] / "Assets" / "datasets" / "skada" / "skada_analytics.sqlite"
        if not db_path.exists():
            return
        conn = sqlite3.connect(db_path)
        try:
            for row in conn.execute("SELECT card_id, name_zh FROM cards"):
                key = str(row[0] or "").strip().upper()
                val = str(row[1] or "").strip()
                if key and val:
                    self.card_zh[key] = val
            for row in conn.execute("SELECT encounter, name_zh FROM encounters"):
                key = str(row[0] or "").strip().upper()
                val = str(row[1] or "").strip()
                if key and val:
                    self.encounter_zh[key] = val
            for row in conn.execute("SELECT relic_id, name_zh FROM relics"):
                key = str(row[0] or "").strip().upper()
                val = str(row[1] or "").strip()
                if key and val:
                    self.relic_zh[key] = val
        finally:
            conn.close()

    def card(self, token: str) -> str:
        key = str(token or "").strip().upper()
        return self.card_zh.get(key) or _pretty_token(token)

    def encounter(self, token: str) -> str:
        key = str(token or "").strip().upper()
        return self.encounter_zh.get(key) or _pretty_token(token)

    def relic(self, token: str) -> str:
        key = str(token or "").strip().upper()
        return self.relic_zh.get(key) or _pretty_token(token)

    def enemy_group(self, token: str) -> str:
        parts = [part for part in str(token or "").split("+") if part]
        if not parts:
            return "未知"
        return " + ".join(self.encounter(part) for part in parts)

    def generic(self, token: str) -> str:
        if not token:
            return "未知"
        upper = str(token).strip().upper()
        if upper in self.card_zh:
            return self.card_zh[upper]
        if upper in self.encounter_zh:
            return self.encounter_zh[upper]
        if upper in self.relic_zh:
            return self.relic_zh[upper]
        known = {
            "monster": "普通战斗",
            "unknown": "事件",
            "rest_site": "火堆",
            "treasure": "宝箱",
            "shop": "商店",
            "elite": "精英",
            "boss": "Boss",
            "skip_card_reward": "跳过卡奖",
            "rest": "休息",
            "smith": "锻造",
            "proceed": "离开/继续",
            "remove_card": "删牌",
        }
        lower = str(token).strip().lower()
        return known.get(lower) or _pretty_token(str(token))


@dataclass
class CombatSnapshot:
    combat_index: int
    floor: int
    start_hp: int
    start_intent: str
    enemy_group: str
    potion_uses: int = 0
    repeat_hits: int = 0
    last_intent: str = ""
    last_action: str = ""
    won: bool = False
    ended_by_death: bool = False


@dataclass
class EpisodeRecord:
    path: Path
    iteration: int
    episode: int
    tag: str
    floor_from_name: int
    outcome: str = "unknown"
    floor: int = 0
    combats: int = 0
    seconds: float = 0.0
    end_reason: str = "unknown"
    error: str = ""
    neow_choice: str = ""
    event_ids: list[str] = field(default_factory=list)
    event_choices: list[str] = field(default_factory=list)
    map_choices: list[tuple[int, str]] = field(default_factory=list)
    card_rewards: list[str] = field(default_factory=list)
    shop_visits: int = 0
    shop_sessions: list[list[str]] = field(default_factory=list)
    rest_sessions: list[list[str]] = field(default_factory=list)
    potion_uses: int = 0
    repeat_max: int = 0
    max_steps_hit: bool = False
    terminal_floor: int = 0
    terminal_hp: int = 0
    terminal_enemy_group: str = ""
    terminal_intent: str = ""
    combats_info: list[CombatSnapshot] = field(default_factory=list)


def _parse_episode(path: Path) -> EpisodeRecord:
    match = FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"无法解析 replay 文件名: {path.name}")
    record = EpisodeRecord(
        path=path,
        iteration=int(match.group("iteration")),
        episode=int(match.group("episode")),
        tag=match.group("tag"),
        floor_from_name=int(match.group("floor")),
    )
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    current_shop: list[str] | None = None
    current_rest: list[str] | None = None
    current_combat: CombatSnapshot | None = None

    def _flush_shop() -> None:
        nonlocal current_shop
        if current_shop is not None:
            record.shop_sessions.append(current_shop)
        current_shop = None

    def _flush_rest() -> None:
        nonlocal current_rest
        if current_rest is not None:
            record.rest_sessions.append(current_rest)
        current_rest = None

    for line in lines:
        if line.startswith("# outcome="):
            header = HEADER_RE.match(line)
            if header is not None:
                record.outcome = header.group("outcome")
                record.floor = int(header.group("floor"))
                record.combats = int(header.group("combats"))
                record.seconds = float(header.group("time"))
                record.end_reason = header.group("end_reason")
                record.error = header.group("error")
            continue

        combat_start = COMBAT_START_RE.match(line)
        if combat_start is not None:
            _flush_shop()
            _flush_rest()
            current_combat = CombatSnapshot(
                combat_index=int(combat_start.group("combat_index")),
                floor=int(combat_start.group("floor")),
                start_hp=int(combat_start.group("hp")),
                start_intent=combat_start.group("intent"),
                enemy_group=_enemy_group(combat_start.group("intent")),
                last_intent=combat_start.group("intent"),
            )
            record.combats_info.append(current_combat)
            continue

        map_match = MAP_RE.match(line)
        if map_match is not None:
            _flush_shop()
            _flush_rest()
            label = map_match.group("label").strip()
            floor = int(map_match.group("floor"))
            record.map_choices.append((floor, label))
            if label == "shop":
                record.shop_visits += 1
                current_shop = []
            elif label == "rest_site":
                current_rest = []
            continue

        card_match = CARD_REWARD_RE.match(line)
        if card_match is not None:
            _flush_shop()
            _flush_rest()
            record.card_rewards.append(card_match.group("label").strip())
            continue

        shop_match = SHOP_RE.match(line)
        if shop_match is not None:
            if current_shop is None:
                current_shop = []
            current_shop.append(shop_match.group("label").strip())
            continue

        rest_match = REST_RE.match(line)
        if rest_match is not None:
            if current_rest is None:
                current_rest = []
            current_rest.append(rest_match.group("label").strip())
            continue

        event_match = EVENT_RE.match(line)
        if event_match is not None:
            _flush_shop()
            _flush_rest()
            event_id = (event_match.group("event_id") or "").strip()
            label = event_match.group("label").strip()
            if event_id:
                record.event_ids.append(event_id)
                if event_id == "EVENT.NEOW" and not record.neow_choice:
                    record.neow_choice = label
            record.event_choices.append(label)
            continue

        combat_nn = COMBAT_NN_RE.match(line)
        if combat_nn is not None and current_combat is not None:
            body = combat_nn.group("body")
            current_combat.last_action = body
            intent_idx = body.find(" intent=[")
            if intent_idx >= 0:
                intent_text = body[intent_idx + len(" intent=[") :]
                current_combat.last_intent = intent_text.split("]", 1)[0]
            if "use_potion" in body:
                record.potion_uses += 1
                current_combat.potion_uses += 1
            continue

        if "COMBAT WON ->" in line and current_combat is not None:
            current_combat.won = True
            _flush_shop()
            _flush_rest()
            continue

        terminal = TERMINAL_RE.match(line)
        if terminal is not None:
            record.terminal_floor = int(terminal.group("floor"))
            record.terminal_hp = int(terminal.group("hp"))
            if current_combat is not None:
                current_combat.ended_by_death = True
                record.terminal_enemy_group = current_combat.enemy_group
                record.terminal_intent = current_combat.last_intent or current_combat.start_intent
            continue

        if "[END] max_steps reached" in line:
            record.max_steps_hit = True
            continue

        repeat_match = REPEAT_RE.search(line)
        if repeat_match is not None:
            repeat_count = int(repeat_match.group("count"))
            record.repeat_max = max(record.repeat_max, repeat_count)
            if current_combat is not None:
                current_combat.repeat_hits = max(current_combat.repeat_hits, repeat_count)
            continue

    _flush_shop()
    _flush_rest()
    return record


def _build_report(
    records: list[EpisodeRecord],
    metrics_entry: dict[str, Any] | None,
    top_k: int,
    resolver: SkadaNameResolver,
) -> dict[str, Any]:
    floors = [record.floor for record in records]
    death_records = [record for record in records if record.outcome == "death"]
    death_floors = [record.floor for record in death_records]
    outcome_counts = Counter(record.outcome for record in records)
    tag_counts = Counter(record.tag for record in records)
    floor_bins = Counter()
    for floor in floors:
        if floor <= 5:
            floor_bins["1-5"] += 1
        elif floor <= 9:
            floor_bins["6-9"] += 1
        elif floor <= 14:
            floor_bins["10-14"] += 1
        else:
            floor_bins["15+"] += 1

    terminal_enemy_counts = Counter(
        record.terminal_enemy_group or "UNKNOWN"
        for record in death_records
    )
    terminal_intent_counts = Counter(
        record.terminal_intent or "UNKNOWN"
        for record in death_records
    )
    map_counts = Counter(label for record in records for _floor, label in record.map_choices)
    early_map_counts = Counter(
        label
        for record in records
        for floor, label in record.map_choices
        if floor <= 8
    )
    neow_counts = Counter(record.neow_choice for record in records if record.neow_choice)
    event_id_counts = Counter(event_id for record in records for event_id in record.event_ids if event_id != "EVENT.NEOW")
    card_reward_counts = Counter(label for record in records for label in record.card_rewards if label != "skip_card_reward")
    card_skip_count = sum(
        1
        for record in records
        for label in record.card_rewards
        if label == "skip_card_reward"
    )
    card_reward_total = sum(len(record.card_rewards) for record in records)
    shop_action_counts = Counter(
        action
        for record in records
        for session in record.shop_sessions
        for action in session
    )
    rest_action_counts = Counter(
        action
        for record in records
        for session in record.rest_sessions
        for action in session
    )
    empty_shop_count = sum(
        1
        for record in records
        for session in record.shop_sessions
        if session == ["proceed"]
    )
    potion_episode_count = sum(1 for record in records if record.potion_uses > 0)
    potion_fight_counts = Counter()
    multi_potion_fights: list[dict[str, Any]] = []
    for record in records:
        for combat in record.combats_info:
            if combat.potion_uses > 0:
                key = f"{combat.enemy_group}@f{combat.floor}"
                potion_fight_counts[key] += combat.potion_uses
                if combat.potion_uses >= 2:
                    multi_potion_fights.append(
                        {
                            "episode": record.episode,
                            "floor": record.floor,
                            "combat_floor": combat.floor,
                            "enemy_group": combat.enemy_group,
                            "enemy_group_display": resolver.enemy_group(combat.enemy_group),
                            "potion_uses": combat.potion_uses,
                            "path": str(record.path.resolve()),
                        }
                    )
    max_step_records = [
        {
            "episode": record.episode,
            "floor": record.floor,
            "path": str(record.path.resolve()),
        }
        for record in records
        if record.max_steps_hit
    ]
    early_deaths = sorted(
        death_records,
        key=lambda record: (record.floor, record.episode),
    )[:top_k]
    suspicious_records = sorted(
        records,
        key=lambda record: (
            record.floor,
            -record.potion_uses,
            -record.repeat_max,
            record.episode,
        ),
    )[:top_k]

    return {
        "episode_count": len(records),
        "metrics": metrics_entry or {},
        "outcomes": dict(sorted(outcome_counts.items())),
        "tags": dict(sorted(tag_counts.items())),
        "floors": {
            "avg": round(_mean([float(value) for value in floors]), 4),
            "median": round(float(median(floors)) if floors else 0.0, 4),
            "min": min(floors) if floors else 0,
            "max": max(floors) if floors else 0,
            "bins": dict(sorted(floor_bins.items())),
            "death_avg": round(_mean([float(value) for value in death_floors]), 4),
            "death_median": round(float(median(death_floors)) if death_floors else 0.0, 4),
        },
        "death": {
            "count": len(death_records),
            "terminal_enemy_top": [
                {
                    "name": name,
                    "display": resolver.enemy_group(name),
                    "count": count,
                }
                for name, count in terminal_enemy_counts.most_common(top_k)
            ],
            "terminal_intent_top": _top(terminal_intent_counts, top_k),
            "early_deaths": [
                {
                    "episode": record.episode,
                    "floor": record.floor,
                    "terminal_enemy_group": record.terminal_enemy_group or "UNKNOWN",
                    "terminal_enemy_display": resolver.enemy_group(record.terminal_enemy_group or "UNKNOWN"),
                    "terminal_intent": record.terminal_intent or "UNKNOWN",
                    "path": str(record.path.resolve()),
                }
                for record in early_deaths
            ],
        },
        "route": {
            "neow_top": _top(neow_counts, top_k),
            "map_top": [
                {"name": name, "display": resolver.generic(name), "count": count}
                for name, count in map_counts.most_common(top_k)
            ],
            "early_map_top": [
                {"name": name, "display": resolver.generic(name), "count": count}
                for name, count in early_map_counts.most_common(top_k)
            ],
            "event_top": _top(event_id_counts, top_k),
        },
        "rewards": {
            "card_reward_total": card_reward_total,
            "card_reward_skip_count": card_skip_count,
            "card_reward_skip_rate": round(_pct(card_skip_count, card_reward_total), 4),
            "card_pick_top": [
                {"name": name, "display": resolver.generic(name), "count": count}
                for name, count in card_reward_counts.most_common(top_k)
            ],
        },
        "shop": {
            "shop_visit_count": sum(record.shop_visits for record in records),
            "empty_shop_count": empty_shop_count,
            "empty_shop_rate": round(
                _pct(empty_shop_count, sum(record.shop_visits for record in records)), 4
            ),
            "shop_action_top": [
                {"name": name, "display": resolver.generic(name), "count": count}
                for name, count in shop_action_counts.most_common(top_k)
            ],
        },
        "rest": {
            "rest_action_top": [
                {"name": name, "display": resolver.generic(name), "count": count}
                for name, count in rest_action_counts.most_common(top_k)
            ],
        },
        "combat": {
            "potion_use_total": sum(record.potion_uses for record in records),
            "potion_use_episode_count": potion_episode_count,
            "potion_use_episode_rate": round(_pct(potion_episode_count, len(records)), 4),
            "potion_fight_top": [
                {
                    "name": name,
                    "display": resolver.enemy_group(name.split("@", 1)[0]) + f" @{name.split('@', 1)[1]}",
                    "count": count,
                }
                for name, count in potion_fight_counts.most_common(top_k)
            ],
            "multi_potion_fights": multi_potion_fights[:top_k],
            "max_step_count": len(max_step_records),
            "max_step_records": max_step_records[:top_k],
        },
        "suspicious_samples": [
            {
                "episode": record.episode,
                "floor": record.floor,
                "outcome": record.outcome,
                "potion_uses": record.potion_uses,
                "repeat_max": record.repeat_max,
                "terminal_enemy_group": record.terminal_enemy_group or "UNKNOWN",
                "terminal_enemy_display": resolver.enemy_group(record.terminal_enemy_group or "UNKNOWN"),
                "path": str(record.path.resolve()),
            }
            for record in suspicious_records
        ],
    }


def _format_top(items: list[dict[str, Any]], *, as_pct_of: float | None = None) -> list[str]:
    lines: list[str] = []
    for item in items:
        label = item.get("display") or item.get("name")
        if as_pct_of is not None and as_pct_of > 0:
            pct = item["count"] / as_pct_of * 100.0
            lines.append(f"- `{label}`: {item['count']} ({pct:.1f}%)")
        else:
            lines.append(f"- `{label}`: {item['count']}")
    return lines or ["- 无"]


def _write_markdown(report: dict[str, Any], output_path: Path) -> None:
    metrics = report.get("metrics") or {}
    floors = report["floors"]
    death = report["death"]
    route = report["route"]
    rewards = report["rewards"]
    shop = report["shop"]
    rest = report["rest"]
    combat = report["combat"]
    episode_count = report["episode_count"]

    lines: list[str] = []
    lines.append(f"# Iter {metrics.get('iteration', 'N/A')} Replay 分析")
    lines.append("")
    lines.append("## 总览")
    lines.append(f"- 样本数: `{episode_count}`")
    if metrics:
        lines.append(f"- `avg_floor`: `{metrics.get('avg_floor', 0):.4f}`")
        lines.append(f"- `boss_reach_rate`: `{metrics.get('boss_reach_rate', 0) * 100:.2f}%`")
        lines.append(f"- `act1_clear_rate`: `{metrics.get('act1_clear_rate', 0) * 100:.2f}%`")
        lines.append(f"- `card_reward_skip_rate`: `{metrics.get('card_reward_skip_rate', 0) * 100:.2f}%`")
        lines.append(f"- `hard_state_potion_steps`: `{metrics.get('hard_state_potion_steps', 0)}`")
        lines.append(f"- `hard_state_premature_end_turn_steps`: `{metrics.get('hard_state_premature_end_turn_steps', 0)}`")
    lines.append(
        f"- 层数分布: 平均 `{floors['avg']:.2f}` / 中位 `{floors['median']:.2f}` / 最低 `{floors['min']}` / 最高 `{floors['max']}`"
    )
    lines.append(f"- 分桶: `{json.dumps(floors['bins'], ensure_ascii=False)}`")
    lines.append("")

    lines.append("## 死亡分析")
    lines.append(f"- 死亡局数: `{death['count']}`")
    lines.append(f"- 死亡层均值/中位: `{floors['death_avg']:.2f}` / `{floors['death_median']:.2f}`")
    lines.extend(_format_top(death["terminal_enemy_top"], as_pct_of=death["count"]))
    lines.append("")
    lines.append("### 终局意图 Top")
    lines.extend(_format_top(death["terminal_intent_top"], as_pct_of=death["count"]))
    lines.append("")
    lines.append("### 最早死亡样本")
    if death["early_deaths"]:
        for sample in death["early_deaths"]:
            lines.append(
                f"- `e{sample['episode']:03d}` floor `{sample['floor']}` "
                f"敌人 `{sample['terminal_enemy_display']}` 意图 `{sample['terminal_intent']}` "
                f"[replay](/" + sample["path"].replace("\\", "/") + ")"
            )
    else:
        lines.append("- 无")
    lines.append("")

    lines.append("## 路线与非战斗")
    lines.append("### Neow 选择 Top")
    lines.extend(_format_top(route["neow_top"], as_pct_of=episode_count))
    lines.append("")
    lines.append("### 地图选择 Top")
    lines.extend(_format_top(route["map_top"]))
    lines.append("")
    lines.append("### 前 8 层地图选择 Top")
    lines.extend(_format_top(route["early_map_top"]))
    lines.append("")
    lines.append("### 事件 Top")
    lines.extend(_format_top(route["event_top"]))
    lines.append("")
    lines.append("### 卡奖")
    lines.append(
        f"- 卡奖总次数: `{rewards['card_reward_total']}`，跳过 `{rewards['card_reward_skip_count']}` "
        f"({rewards['card_reward_skip_rate'] * 100:.2f}%)"
    )
    lines.extend(_format_top(rewards["card_pick_top"]))
    lines.append("")
    lines.append("### 商店")
    lines.append(
        f"- 进店次数: `{shop['shop_visit_count']}`，空店直接走人: `{shop['empty_shop_count']}` "
        f"({shop['empty_shop_rate'] * 100:.2f}%)"
    )
    lines.extend(_format_top(shop["shop_action_top"]))
    lines.append("")
    lines.append("### 火堆")
    lines.extend(_format_top(rest["rest_action_top"]))
    lines.append("")

    lines.append("## 战斗行为")
    lines.append(
        f"- 药水使用总次数: `{combat['potion_use_total']}`，发生在 `{combat['potion_use_episode_count']}` 把 "
        f"({combat['potion_use_episode_rate'] * 100:.2f}%)"
    )
    lines.append("### 药水高发战斗 Top")
    lines.extend(_format_top(combat["potion_fight_top"]))
    lines.append("")
    lines.append("### 单场多次用药")
    if combat["multi_potion_fights"]:
        for sample in combat["multi_potion_fights"]:
            lines.append(
                f"- `e{sample['episode']:03d}` floor `{sample['floor']}` combat_floor `{sample['combat_floor']}` "
                f"敌人 `{sample['enemy_group_display']}` 用药 `{sample['potion_uses']}` 次 "
                f"[replay](/" + sample["path"].replace("\\", "/") + ")"
            )
    else:
        lines.append("- 无")
    lines.append("")
    lines.append("### Max Steps")
    lines.append(f"- `max_steps` 局数: `{combat['max_step_count']}`")
    if combat["max_step_records"]:
        for sample in combat["max_step_records"]:
            lines.append(
                f"- `e{sample['episode']:03d}` floor `{sample['floor']}` "
                f"[replay](/" + sample["path"].replace("\\", "/") + ")"
            )
    else:
        lines.append("- 无")
    lines.append("")

    lines.append("## 可疑样本")
    for sample in report["suspicious_samples"]:
        lines.append(
            f"- `e{sample['episode']:03d}` outcome `{sample['outcome']}` floor `{sample['floor']}` "
            f"药水 `{sample['potion_uses']}` repeat `{sample['repeat_max']}` "
            f"终局敌人 `{sample['terminal_enemy_display']}` "
            f"[replay](/" + sample["path"].replace("\\", "/") + ")"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="按 iteration 分析 hybrid 训练 replay。")
    parser.add_argument("training_dir", help="训练输出目录，例如 STS2AI/Artifacts/.../20260414-110316_4env_acttransitionfix_resume2275")
    parser.add_argument("--iteration", type=int, required=True, help="要分析的 iteration")
    parser.add_argument("--top-k", type=int, default=10, help="报告里保留的 Top 数量")
    args = parser.parse_args()

    training_dir = Path(args.training_dir)
    replay_dir = training_dir / "replays"
    metrics_path = training_dir / "metrics.jsonl"
    output_dir = training_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    replay_files = sorted(replay_dir.glob(f"i{args.iteration:05d}_*.txt"))
    if not replay_files:
        raise SystemExit(f"未找到 iteration {args.iteration} 的 replay 文件: {replay_dir}")

    records = [_parse_episode(path) for path in replay_files]
    metrics_entry = _load_metrics_entry(metrics_path, args.iteration)
    resolver = SkadaNameResolver()
    report = _build_report(records, metrics_entry, args.top_k, resolver)

    json_path = output_dir / f"iter_{args.iteration:05d}_replay_report.json"
    md_path = output_dir / f"iter_{args.iteration:05d}_replay_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(report, md_path)

    print(f"JSON: {json_path}")
    print(f"MD: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
