#!/usr/bin/env python3
"""Build non-combat SFT rows from Skada full-detail run records.

This intentionally uses only choices with visible human labels in Skada detail:
card rewards, relic choices, campfire choices, shop actions, and map transitions.
It does not infer combat action labels.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

_STS2AI_ROOT = Path(__file__).resolve().parents[2]
if str(_STS2AI_ROOT) not in sys.path:
    sys.path.insert(0, str(_STS2AI_ROOT))

from llm.data_pipeline.card_effects import (  # noqa: E402
    card_is_power,
    effective_block,
    effective_damage,
    lookup_effect,
)
from llm.data_pipeline.catalog_loader import render_card_description  # noqa: E402
from llm.prompts import load_system_prompt  # noqa: E402


_REPO_ROOT = _STS2AI_ROOT.parent
_CATALOG_DB = _STS2AI_ROOT / "data" / "game_wiki" / "game_catalog.sqlite"
_ENG_RELICS_JSON = _REPO_ROOT / "localization" / "eng" / "relics.json"
_MARKUP_RE = re.compile(r"\[/?(?:gold|yellow|red|blue|green|cyan|magenta|white|orange|purple)\]")
_IMG_TAG_RE = re.compile(r"\[img\].*?\[/img\]")
ROOM_NAMES = {
    "A": "Act Start",
    "B": "Boss",
    "E": "Elite",
    "M": "Monster",
    "R": "Rest Site",
    "S": "Shop",
    "T": "Treasure",
    "V": "Event",
    "?": "Unknown",
}

STARTER_DECKS = {
    "IRONCLAD": [
        ("STRIKE_IRONCLAD", 5),
        ("DEFEND_IRONCLAD", 4),
        ("BASH", 1),
    ],
    "SILENT": [
        ("STRIKE_SILENT", 5),
        ("DEFEND_SILENT", 5),
        ("NEUTRALIZE", 1),
        ("SURVIVOR", 1),
    ],
    "DEFECT": [
        ("STRIKE_DEFECT", 4),
        ("DEFEND_DEFECT", 4),
        ("ZAP", 1),
        ("DUALCAST", 1),
    ],
}

STARTER_RELICS = {
    "IRONCLAD": ["BURNING_BLOOD"],
    "SILENT": ["RING_OF_THE_SNAKE", "RING_OF_THE_DRAKE"],
    "DEFECT": ["CRACKED_CORE"],
}

DRAW_CARDS = {
    "POMMEL_STRIKE", "POMMEL_STRIKE+", "SHRUG_IT_OFF", "SHRUG_IT_OFF+",
    "BATTLE_TRANCE", "BATTLE_TRANCE+", "OFFERING", "OFFERING+",
    "BURNING_PACT", "BURNING_PACT+", "DARK_EMBRACE", "DARK_EMBRACE+",
    "FINESSE", "FINESSE+", "EXHUME", "EXHUME+",
}
ENERGY_CARDS = {
    "BLOODLETTING", "BLOODLETTING+", "OFFERING", "OFFERING+",
    "SEEING_RED", "SEEING_RED+", "FORGOTTEN_RITUAL", "FORGOTTEN_RITUAL+",
}
EXHAUST_CARDS = {
    "TRUE_GRIT", "TRUE_GRIT+", "HAVOC", "HAVOC+", "BURNING_PACT", "BURNING_PACT+",
    "FEEL_NO_PAIN", "FEEL_NO_PAIN+", "DARK_EMBRACE", "DARK_EMBRACE+",
    "SECOND_WIND", "SECOND_WIND+", "FIEND_FIRE", "FIEND_FIRE+", "EXHUME", "EXHUME+",
}
STRENGTH_CARDS = {
    "INFLAME", "INFLAME+", "DEMON_FORM", "DEMON_FORM+", "FLEX", "FLEX+",
    "SPOT_WEAKNESS", "SPOT_WEAKNESS+", "LIMIT_BREAK", "LIMIT_BREAK+", "RUPTURE", "RUPTURE+",
}
VULNERABLE_CARDS = {"BASH", "BASH+", "THUNDERCLAP", "THUNDERCLAP+", "UPPERCUT", "UPPERCUT+", "SHOCKWAVE", "SHOCKWAVE+"}
AOE_CARDS = {"CLEAVE", "CLEAVE+", "THUNDERCLAP", "THUNDERCLAP+", "WHIRLWIND", "WHIRLWIND+", "IMMOLATE", "IMMOLATE+"}
BLOCK_SCALING_CARDS = {"BODY_SLAM", "BODY_SLAM+", "BARRICADE", "BARRICADE+", "ENTRENCH", "ENTRENCH+", "JUGGERNAUT", "JUGGERNAUT+"}
WEAK_STARTER_CARDS = {
    "STRIKE_IRONCLAD", "STRIKE_IRONCLAD+", "DEFEND_IRONCLAD", "DEFEND_IRONCLAD+",
    "STRIKE_SILENT", "STRIKE_SILENT+", "DEFEND_SILENT", "DEFEND_SILENT+",
    "STRIKE_DEFECT", "STRIKE_DEFECT+", "DEFEND_DEFECT", "DEFEND_DEFECT+",
}


@dataclass
class BuildState:
    character: str
    hp: int = 0
    max_hp: int = 0
    gold: int = 99
    deck: Counter[str] = field(default_factory=Counter)
    relics: list[str] = field(default_factory=list)

    @classmethod
    def starter(cls, character: str) -> "BuildState":
        normalized = character.upper()
        deck: Counter[str] = Counter()
        for card_id, count in STARTER_DECKS.get(normalized, []):
            deck[card_id] += count
        return cls(
            character=normalized,
            deck=deck,
            relics=list(STARTER_RELICS.get(normalized, [])),
        )

    def update_resource_snapshot(self, row: dict[str, Any]) -> None:
        self.hp = int(row.get("hp_before") or self.hp or 0)
        self.max_hp = max(int(self.max_hp or 0), self.hp, int(row.get("hp_after") or 0), 1)
        self.gold = int(row.get("gold_before") or self.gold or 0)

    def apply_floor_result(self, row: dict[str, Any]) -> None:
        for choice in _as_list(row.get("relic_choices")):
            if choice.get("was_picked"):
                relic_id = _clean_id(choice.get("relic_id"))
                if relic_id and relic_id not in self.relics:
                    self.relics.append(relic_id)
        for choice in _as_list(row.get("card_choices")):
            if choice.get("was_picked"):
                card_id = _clean_id(choice.get("card_id"))
                if card_id:
                    self.deck[card_id] += 1
        for upgrade in _as_list(row.get("card_upgrades")):
            card_id = _clean_id(upgrade.get("card_id"))
            if not card_id:
                continue
            base = card_id.rstrip("+")
            if self.deck[base] > 0:
                self.deck[base] -= 1
                if self.deck[base] <= 0:
                    del self.deck[base]
            self.deck[card_id] += 1
        for action in _as_list(row.get("shop_actions")):
            action_type = str(action.get("action_type") or "").lower()
            item_id = _clean_id(action.get("item_id"))
            if not item_id:
                continue
            if action_type == "remove":
                if self.deck[item_id] > 0:
                    self.deck[item_id] -= 1
                    if self.deck[item_id] <= 0:
                        del self.deck[item_id]
            elif action_type == "buy_card":
                self.deck[item_id] += 1
            elif action_type == "buy_relic" and item_id not in self.relics:
                self.relics.append(item_id)
        self.hp = int(row.get("hp_after") or self.hp or 0)
        self.gold = int(row.get("gold_after") or self.gold or 0)

    def deck_line(self, *, limit: int = 40) -> str:
        if not self.deck:
            return "-"
        parts = [f"{card}x{count}" if count > 1 else card for card, count in sorted(self.deck.items())]
        if len(parts) > limit:
            return ", ".join(parts[:limit]) + f", ...(+{len(parts) - limit})"
        return ", ".join(parts)

    def relic_line(self, *, limit: int = 16, include_descriptions: bool = False) -> str:
        if not self.relics:
            return "-"
        parts = [
            _relic_line(relic, include_description=include_descriptions)
            if include_descriptions else relic
            for relic in self.relics
        ]
        if len(parts) > limit:
            return ", ".join(parts[:limit]) + f", ...(+{len(parts) - limit})"
        return ", ".join(parts)


@dataclass(frozen=True)
class BuildProfile:
    deck_size: int
    upgraded: int
    attacks: int
    block_cards: int
    powers: int
    draw: int
    energy: int
    exhaust: int
    vulnerable: int
    strength: int
    aoe: int
    block_scaling: int
    key_cards: list[str]
    key_relics: list[str]

    def archetype(self) -> str:
        if self.exhaust >= 2 or any(card in self.key_cards for card in ("FEEL_NO_PAIN", "DARK_EMBRACE")):
            return "exhaust value"
        if self.strength >= 2 or any(card in self.key_cards for card in ("INFLAME", "DEMON_FORM", "LIMIT_BREAK")):
            return "strength-scaling attacks"
        if self.block_scaling >= 1 and self.block_cards >= 5:
            return "block scaling"
        if self.vulnerable >= 1 and self.attacks >= max(5, self.block_cards):
            return "vulnerable attack tempo"
        if self.deck_size <= 13:
            return "lean efficient deck"
        return "flexible balanced deck"

    def needs(self, *, floor: int = 0) -> list[str]:
        needs: list[str] = []
        if self.attacks <= 5:
            needs.append("efficient damage")
        if self.block_cards <= 4:
            needs.append("reliable block")
        if self.draw <= 1 and self.deck_size >= 13:
            needs.append("card draw")
        if self.aoe == 0 and floor >= 6:
            needs.append("area damage")
        if self.energy == 0 and self.deck_size >= 16:
            needs.append("energy support")
        if self.powers == 0 and floor >= 10:
            needs.append("scaling")
        if len(needs) < 2 and self.deck_size >= 18:
            needs.append("deck quality")
        return needs[:4]

    def summary_line(self) -> str:
        return (
            f"build_summary: deck_size={self.deck_size} upgraded={self.upgraded} "
            f"attacks={self.attacks} block={self.block_cards} powers={self.powers} "
            f"draw={self.draw} energy={self.energy} exhaust={self.exhaust} "
            f"vulnerable={self.vulnerable} strength={self.strength} aoe={self.aoe}"
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--details-root", type=Path, default=_STS2AI_ROOT / "data" / "skada" / "runs_full_detail")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--include-failure", action="store_true")
    parser.add_argument("--character", default="", help="Optional character filter, for example IRONCLAD.")
    parser.add_argument("--game-version", default="", help="Optional exact game_version filter.")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--max-per-type", type=int, default=20000)
    parser.add_argument(
        "--max-per-type-overrides",
        default="",
        help=(
            "Comma separated per decision cap overrides, for example "
            "card_reward=10000,map_choice=2000. Types not listed use --max-per-type."
        ),
    )
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument(
        "--teacher-style",
        choices=("strategic_v2", "human_match"),
        default="strategic_v2",
        help="strategic_v2 adds visible build summaries and plan/reason labels; human_match keeps the old minimal labels.",
    )
    return parser.parse_args()


def _parse_type_cap_overrides(raw: str) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"invalid --max-per-type-overrides entry: {part!r}")
        key, value = part.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"invalid empty decision type in override: {part!r}")
        overrides[key] = int(value.strip())
    return overrides


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _clean_id(value: Any) -> str:
    return str(value or "").strip().upper()


def _display(item: dict[str, Any], id_key: str) -> str:
    display = item.get("display_name")
    if isinstance(display, dict):
        text = str(display.get("en") or display.get("zh") or "").strip()
        if text:
            return text
    return _clean_id(item.get(id_key))


def _room_name(room_type: Any) -> str:
    key = str(room_type or "?").upper()
    return ROOM_NAMES.get(key, key)


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = _IMG_TAG_RE.sub("", text)
    text = _MARKUP_RE.sub("", text)
    text = text.replace("콘좆", "Energy")
    text = text.replace("\\n", " ").replace("\n", " ").replace("；", ";")
    return " ".join(text.split()).strip()


def _card_base(card_id: str) -> str:
    return _clean_id(card_id).rstrip("+")


def _card_upgraded(card_id: str) -> bool:
    return _clean_id(card_id).endswith("+")


@lru_cache(maxsize=1)
def _card_catalog() -> dict[str, dict[str, Any]]:
    if not _CATALOG_DB.exists():
        return {}
    conn = sqlite3.connect(str(_CATALOG_DB))
    try:
        rows = conn.execute(
            "SELECT id, name_en, description_en, card_type, rarity, base_cost, target_type, tags_json, keywords_json "
            "FROM cards"
        ).fetchall()
    finally:
        conn.close()
    out: dict[str, dict[str, Any]] = {}
    for cid, name, desc, card_type, rarity, cost, target, tags_json, keywords_json in rows:
        key = _clean_id(cid)
        if not key:
            continue
        out[key] = {
            "id": key,
            "name": name or key,
            "description": desc or "",
            "card_type": str(card_type or "").lower(),
            "rarity": str(rarity or "").lower(),
            "cost": cost if cost is not None else "?",
            "target_type": str(target or "").lower(),
            "tags": _load_json_list(tags_json),
            "keywords": _load_json_list(keywords_json),
        }
    return out


def _load_json_list(raw: Any) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


def _card_info(card_id: str) -> dict[str, Any]:
    base = _card_base(card_id)
    info = dict(_card_catalog().get(base) or {})
    if not info:
        info = {"id": base, "name": base, "description": "", "card_type": "", "rarity": "", "cost": "?", "target_type": ""}
    return info


def _card_description(card_id: str) -> str:
    info = _card_info(card_id)
    desc = str(info.get("description") or "")
    if desc:
        rendered = render_card_description(_card_base(card_id), desc, is_upgraded=_card_upgraded(card_id))
        rendered = _clean_text(rendered)
        if rendered:
            return rendered
    return ""


@lru_cache(maxsize=1)
def _relic_texts() -> dict[str, dict[str, str]]:
    if not _ENG_RELICS_JSON.exists():
        return {}
    try:
        raw = json.loads(_ENG_RELICS_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, dict[str, str]] = {}
    for key, value in raw.items():
        relic_id, _, field_name = str(key).partition(".")
        if not relic_id or not field_name:
            continue
        out.setdefault(relic_id.upper(), {})[field_name] = str(value or "")
    return out


def _relic_line(relic_id: str, display: str | None = None, *, include_description: bool = True) -> str:
    rid = _clean_id(relic_id)
    texts = _relic_texts().get(rid) or {}
    name = display or str(texts.get("title") or rid).strip()
    desc = str(texts.get("description") or "").replace("\\n", " ").replace("\n", " ").strip()
    if not include_description:
        desc = ""
    return f"{rid} | {name}" + (f" | {desc}" if desc else "")


def _profile_from(deck: Counter[str], relics: list[str]) -> BuildProfile:
    deck_size = sum(deck.values())
    upgraded = sum(count for cid, count in deck.items() if _card_upgraded(cid))
    attacks = block_cards = powers = draw = energy = exhaust = vulnerable = strength = aoe = block_scaling = 0
    key_cards: list[str] = []
    for cid, count in deck.items():
        base = _card_base(cid)
        info = _card_info(base)
        ctype = str(info.get("card_type") or "").lower()
        dmg = effective_damage(base, _card_upgraded(cid))
        block = effective_block(base, _card_upgraded(cid))
        effect = lookup_effect(base, _card_upgraded(cid))
        if ctype == "attack" or dmg > 0:
            attacks += count
        if block > 0:
            block_cards += count
        if ctype == "power" or card_is_power(base):
            powers += count
        if cid in DRAW_CARDS or base in {card.rstrip("+") for card in DRAW_CARDS}:
            draw += count
        if cid in ENERGY_CARDS or base in {card.rstrip("+") for card in ENERGY_CARDS}:
            energy += count
        if cid in EXHAUST_CARDS or base in {card.rstrip("+") for card in EXHAUST_CARDS} or "Exhaust" in info.get("keywords", []):
            exhaust += count
        if cid in VULNERABLE_CARDS or base in {card.rstrip("+") for card in VULNERABLE_CARDS} or effect.applies_vulnerable:
            vulnerable += count
        if cid in STRENGTH_CARDS or base in {card.rstrip("+") for card in STRENGTH_CARDS}:
            strength += count
        if cid in AOE_CARDS or base in {card.rstrip("+") for card in AOE_CARDS} or effect.is_aoe:
            aoe += count
        if cid in BLOCK_SCALING_CARDS or base in {card.rstrip("+") for card in BLOCK_SCALING_CARDS}:
            block_scaling += count
        if (
            ctype == "power"
            or base in {"BASH", "POMMEL_STRIKE", "SHRUG_IT_OFF", "FEEL_NO_PAIN", "DARK_EMBRACE", "INFLAME", "DEMON_FORM", "OFFERING", "BLOODLETTING"}
            or count > 1 and base not in {"STRIKE_IRONCLAD", "DEFEND_IRONCLAD"}
        ):
            key_cards.append(cid)
    key_relics = [
        relic for relic in relics
        if relic in {"BURNING_BLOOD", "RUNIC_PYRAMID", "LANTERN", "BAG_OF_PREPARATION", "VAJRA", "RED_SKULL", "MEMBERSHIP_CARD", "ETERNAL_FEATHER", "ORICHALCUM"}
    ]
    return BuildProfile(
        deck_size=deck_size,
        upgraded=upgraded,
        attacks=attacks,
        block_cards=block_cards,
        powers=powers,
        draw=draw,
        energy=energy,
        exhaust=exhaust,
        vulnerable=vulnerable,
        strength=strength,
        aoe=aoe,
        block_scaling=block_scaling,
        key_cards=list(dict.fromkeys(key_cards))[:8],
        key_relics=key_relics[:6],
    )


def _final_profile(record: dict[str, Any]) -> BuildProfile:
    deck: Counter[str] = Counter()
    for entry in _as_list(record.get("final_deck")):
        if isinstance(entry, dict):
            cid = _clean_id(entry.get("card_id"))
            if cid:
                deck[cid] += int(entry.get("count") or 1)
    relics = [
        _clean_id(entry.get("relic_id"))
        for entry in _as_list(record.get("final_relics"))
        if isinstance(entry, dict) and _clean_id(entry.get("relic_id"))
    ]
    return _profile_from(deck, relics)


def _build_context_lines(record: dict[str, Any], build: BuildState, floor: dict[str, Any]) -> list[str]:
    floor_no = int(floor.get("floor") or 0)
    profile = _profile_from(build.deck, build.relics)
    needs = profile.needs(floor=floor_no)
    priorities = needs or ["upgrade key cards", "avoid low-impact cards"]
    keep = [f"add {need}" for need in priorities[:3]]
    avoid = ["extra weak starter cards", "off-plan expensive cards", "deck bloat"]
    hp_ratio = build.hp / max(1, build.max_hp)
    if hp_ratio < 0.45:
        map_policy = "prioritize rest-safe paths and avoid optional elites"
    elif hp_ratio > 0.75:
        map_policy = "can take elites if the route has a recovery exit"
    else:
        map_policy = "balance fights, upgrades, and shops by current resources"
    lines = [
        profile.summary_line(),
        "current_plan:",
        f"  core: {profile.archetype()}",
        f"  win_condition: convert {', '.join(profile.key_cards[:3]) if profile.key_cards else 'starter cards'} into a coherent deck with enough defense and draw",
        "  priorities: " + ", ".join(keep),
        "  pick_policy: take cards/relics that solve priorities or strengthen the core; skip weak off-plan cards",
        "  remove_policy: remove weak starter/curse/status cards before marginal upgrades",
        f"  route_policy: {map_policy}",
        "needs: " + (", ".join(needs) if needs else "upgrade quality and avoid low-impact cards"),
        "key_cards: " + (", ".join(profile.key_cards) if profile.key_cards else "none yet"),
        "key_relics: " + (", ".join(profile.key_relics) if profile.key_relics else "none yet"),
        "avoid: " + ", ".join(avoid),
    ]
    return lines


def _teacher_plan(record: dict[str, Any], build: BuildState, floor: dict[str, Any]) -> str:
    current = _profile_from(build.deck, build.relics)
    needs = current.needs(floor=int(floor.get("floor") or 0)) or ["deck quality"]
    return f"Current plan: {current.archetype()}. Short-term priorities are {', '.join(needs[:3])}."


def _act_index_for_floor(floor_no: int) -> int:
    if floor_no <= 0:
        return 0
    return max(0, (floor_no - 1) // 17)


def _act_floor_bounds(floor_no: int) -> tuple[int, int]:
    act = _act_index_for_floor(floor_no)
    start = act * 17 + 1
    return start, start + 16


def _floor_no(row: dict[str, Any]) -> int:
    try:
        return int(row.get("floor") or 0)
    except (TypeError, ValueError):
        return 0


def _floor_timeline(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row for row in _as_list(record.get("floor_timeline"))
        if isinstance(row, dict)
    ]


def _boss_context_line(record: dict[str, Any], floor_no: int) -> str:
    start, end = _act_floor_bounds(floor_no)
    for row in _floor_timeline(record):
        if not (start <= _floor_no(row) <= end):
            continue
        if str(row.get("room_type") or "").upper() != "B":
            continue
        combat = row.get("combat") if isinstance(row.get("combat"), dict) else {}
        enc = _clean_id(combat.get("encounter")) or "UNKNOWN"
        display = combat.get("encounter_display_name")
        name = ""
        if isinstance(display, dict):
            name = str(display.get("en") or display.get("zh") or "").strip()
        return f"boss: floor={row.get('floor')} encounter={enc}" + (f" name={name}" if name else "")
    return "boss: unknown"


def _route_ahead_line(record: dict[str, Any], floor_no: int, *, limit: int = 8) -> str:
    act = _act_index_for_floor(floor_no)
    local_floor = floor_no - act * 17
    act_maps = [
        act_map for act_map in _as_list(record.get("map_acts"))
        if isinstance(act_map, dict) and int(act_map.get("act") or -1) == act
    ]
    if not act_maps:
        return "route_ahead: unknown"
    act_map = act_maps[0]
    visited = [coord for coord in _as_list(act_map.get("visited_coords")) if isinstance(coord, list) and len(coord) >= 2]
    nodes = {
        tuple(node.get("coord") or []): node
        for node in _as_list(act_map.get("nodes"))
        if isinstance(node, dict) and isinstance(node.get("coord"), list)
    }
    current_idx = max(0, min(len(visited) - 1, local_floor - 1)) if visited else 0
    future = visited[current_idx + 1: current_idx + 1 + limit]
    parts: list[str] = []
    for offset, coord in enumerate(future, 1):
        node = nodes.get(tuple(coord)) or {}
        parts.append(f"f+{offset}:{_room_name(node.get('type'))}@{list(coord)}")
    boss = act_map.get("boss")
    suffix = f"; boss_coord={boss}" if boss else ""
    return "route_ahead: " + (", ".join(parts) if parts else "none") + suffix


def _recent_combat_line(record: dict[str, Any], floor_no: int, *, limit: int = 3) -> str:
    rows = [
        row for row in _floor_timeline(record)
        if _floor_no(row) < floor_no and isinstance(row.get("combat"), dict)
    ][-limit:]
    if not rows:
        return "recent_combats: none"
    parts: list[str] = []
    for row in rows:
        combat = row.get("combat") if isinstance(row.get("combat"), dict) else {}
        hp_before = int(row.get("hp_before") or 0)
        hp_after = int(row.get("hp_after") or 0)
        parts.append(
            f"f{row.get('floor')}:{_clean_id(combat.get('encounter'))} "
            f"turns={combat.get('turns')} hp_delta={hp_after - hp_before} "
            f"dmg_taken={combat.get('total_dmg_taken')}"
        )
    return "recent_combats: " + "; ".join(parts)


def _next_risk_line(record: dict[str, Any], floor_no: int, *, limit: int = 6) -> str:
    future = [
        row for row in _floor_timeline(record)
        if _floor_no(row) > floor_no
    ][:limit]
    if not future:
        return "next_risk: none"
    parts: list[str] = []
    elite_before_rest = False
    seen_rest = False
    for row in future:
        room = _room_name(row.get("room_type"))
        if room == "Rest Site":
            seen_rest = True
        if room == "Elite" and not seen_rest:
            elite_before_rest = True
        combat = row.get("combat") if isinstance(row.get("combat"), dict) else {}
        enc = _clean_id(combat.get("encounter"))
        label = f"f{row.get('floor')}:{room}"
        if enc:
            label += f"/{enc}"
        parts.append(label)
    return f"next_risk: elite_before_rest={str(elite_before_rest).lower()}; " + ", ".join(parts)


def _long_context_lines(record: dict[str, Any], floor: dict[str, Any]) -> list[str]:
    floor_no = int(floor.get("floor") or 0)
    return [
        _boss_context_line(record, floor_no),
        _route_ahead_line(record, floor_no),
        _recent_combat_line(record, floor_no),
        _next_risk_line(record, floor_no),
        _final_outcome_line(record),
    ]


def _final_outcome_line(record: dict[str, Any]) -> str:
    final = _final_profile(record)
    deck_cards: list[str] = []
    for entry in _as_list(record.get("final_deck")):
        if not isinstance(entry, dict):
            continue
        cid = _clean_id(entry.get("card_id"))
        if not cid:
            continue
        count = int(entry.get("count") or 1)
        deck_cards.append(f"{cid}x{count}" if count > 1 else cid)
        if len(deck_cards) >= 12:
            break
    return (
        f"winning_outcome_reference: final_archetype={final.archetype()} "
        f"final_key_cards={','.join(final.key_cards[:6]) if final.key_cards else 'none'} "
        f"final_deck_head={','.join(deck_cards) if deck_cards else 'unknown'}"
    )


def _candidate_card_line(index: int, action: str, card_id: str, display: str) -> str:
    info = _card_info(card_id)
    attrs = [
        f"type={info.get('card_type') or '?'}",
        f"rarity={info.get('rarity') or '?'}",
        f"cost={info.get('cost')}",
    ]
    desc = _card_description(card_id)
    suffix = f" | {desc}" if desc else ""
    return f"  [{index}] {action} {_clean_id(card_id)} | {display} {' '.join(attrs)}{suffix}"


def _role_for_card(card_id: str) -> str:
    cid = _clean_id(card_id)
    base = _card_base(cid)
    info = _card_info(base)
    if cid in DRAW_CARDS or base in {card.rstrip("+") for card in DRAW_CARDS}:
        return "card draw"
    if cid in ENERGY_CARDS or base in {card.rstrip("+") for card in ENERGY_CARDS}:
        return "energy support"
    if cid in EXHAUST_CARDS or base in {card.rstrip("+") for card in EXHAUST_CARDS}:
        return "exhaust synergy"
    if cid in STRENGTH_CARDS or base in {card.rstrip("+") for card in STRENGTH_CARDS}:
        return "strength scaling"
    if cid in AOE_CARDS or base in {card.rstrip("+") for card in AOE_CARDS}:
        return "area damage"
    if cid in VULNERABLE_CARDS or base in {card.rstrip("+") for card in VULNERABLE_CARDS}:
        return "vulnerable setup"
    if effective_block(base, _card_upgraded(cid)) > 0:
        return "block"
    if effective_damage(base, _card_upgraded(cid)) > 0 or str(info.get("card_type") or "") == "attack":
        return "damage"
    if str(info.get("card_type") or "") == "power":
        return "scaling"
    return "deck utility"


def _selected_reason(decision_type: str, selected: str, record: dict[str, Any], build: BuildState, floor: dict[str, Any]) -> str:
    profile = _profile_from(build.deck, build.relics)
    needs = profile.needs(floor=int(floor.get("floor") or 0))
    need_text = ", ".join(needs[:2]) if needs else "deck quality"
    selected_id = _clean_id(selected.split()[-1] if " " in selected else selected)
    if decision_type == "card_reward":
        if selected_id == "SKIP":
            return f"Skip preserves deck quality because current needs are not solved by the offered cards."
        return f"{selected_id} adds {_role_for_card(selected_id)} and supports {profile.archetype()}; current priorities are {need_text}."
    if decision_type == "relic_select":
        return f"{selected_id} is the strongest long-term relic for the current {profile.archetype()} plan."
    if decision_type == "rest_site_choice":
        hp_ratio = build.hp / max(1, build.max_hp)
        if selected.upper().startswith("REST"):
            return f"Rest protects the run because HP is only {hp_ratio:.0%} before upcoming fights."
        return f"Smith is safe at {hp_ratio:.0%} HP and upgrades the deck's main plan."
    if decision_type == "shop_choice":
        if selected.startswith("remove "):
            target = _clean_id(selected.removeprefix("remove "))
            if target in WEAK_STARTER_CARDS or "STRIKE" in target or "DEFEND" in target:
                return f"Removing {target} improves draw quality for the current {profile.archetype()} plan."
            return f"Removing {target} cuts a low-value card and improves deck consistency."
        if selected.startswith("buy_card "):
            cid = _clean_id(selected.removeprefix("buy_card "))
            return f"Buying {cid} adds {_role_for_card(cid)} that fits {profile.archetype()}."
        if selected.startswith("buy_relic "):
            return f"The relic purchase is higher long-term value than another mediocre card."
        return "Leaving preserves gold because the visible shop options are not essential."
    if decision_type == "map_choice":
        if "Elite" in selected:
            return "Taking the elite path converts current strength into relic value."
        if "Rest Site" in selected:
            return "Pathing to a rest site manages HP and upgrade timing."
        if "Shop" in selected:
            return "Shop path converts gold into deck quality."
        return "This route keeps the run progressing with acceptable risk."
    return f"{selected_id} best supports the current plan."


def _action_scores(selected_index: int, count: int, reason: str) -> list[dict[str, Any]]:
    scores = [{"action_index": int(selected_index), "score": 9.0, "note": reason[:80]}]
    for idx in range(count):
        if idx == selected_index:
            continue
        scores.append({"action_index": int(idx), "score": 5.0, "note": "plausible but lower priority"})
        if len(scores) >= 3:
            break
    return scores


def _iter_records(root: Path, *, include_failure: bool) -> Iterable[tuple[Path, int, str, dict[str, Any]]]:
    splits = ["victory"] + (["failure"] if include_failure else [])
    for split in splits:
        for path in sorted((root / split / "details").glob("run_details_*.jsonl")):
            with path.open("r", encoding="utf-8-sig") as handle:
                for line_no, line in enumerate(handle, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        yield path, line_no, split, payload


def _base_lines(
    record: dict[str, Any],
    build: BuildState,
    row: dict[str, Any],
    decision_type: str,
    *,
    include_strategy: bool,
) -> list[str]:
    run = record.get("run") or {}
    lines = [
        f"decision_type: {decision_type}",
        (
            f"run: character={run.get('character')} ascension={run.get('ascension')} "
            f"version={run.get('game_version')} victory={bool(run.get('is_victory'))}"
        ),
        (
            f"floor: {row.get('floor')} room={_room_name(row.get('room_type'))} "
            f"hp={build.hp}/{build.max_hp or '?'} gold={build.gold}"
        ),
        f"deck: {build.deck_line()}",
        f"relics: {build.relic_line(include_descriptions=include_strategy)}",
    ]
    if include_strategy:
        lines.extend(_long_context_lines(record, row))
        lines.extend(_build_context_lines(record, build, row))
    return lines


def _assistant(action_index: int, reason: str, *, confidence: float = 0.9, plan: str = "", action_count: int = 1) -> str:
    payload = {
        "action_index": int(action_index),
        "confidence": round(float(confidence), 2),
        "action_scores": _action_scores(action_index, max(1, action_count), reason),
        "reason": reason[:160],
    }
    if plan:
        payload["plan"] = plan[:180]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _row(system_prompt: str, user: str, assistant: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "meta": meta,
    }


def _card_reward_rows(
    path: Path,
    line_no: int,
    split: str,
    record: dict[str, Any],
    build: BuildState,
    floor: dict[str, Any],
    system_prompt: str,
    teacher_style: str,
) -> list[dict[str, Any]]:
    choices = [c for c in _as_list(floor.get("card_choices")) if isinstance(c, dict)]
    if not choices:
        return []
    picked_index = next((i for i, choice in enumerate(choices) if choice.get("was_picked")), len(choices))
    selected = "SKIP" if picked_index == len(choices) else _clean_id(choices[picked_index].get("card_id"))
    lines = _base_lines(record, build, floor, "card_reward", include_strategy=teacher_style == "strategic_v2")
    lines.append("legal_actions:")
    for idx, choice in enumerate(choices):
        cid = _clean_id(choice.get("card_id"))
        if teacher_style == "strategic_v2":
            lines.append(_candidate_card_line(idx, "pick_card", cid, _display(choice, "card_id")))
        else:
            lines.append(f"  [{idx}] pick_card {cid} | {_display(choice, 'card_id')}")
    lines.append(f"  [{len(choices)}] skip")
    if teacher_style == "strategic_v2":
        lines.append('Return one JSON line: {"action_index":N,"confidence":0.0,"action_scores":[{"action_index":N,"score":0.0}],"plan":"...","reason":"..."}')
        reason = _selected_reason("card_reward", selected, record, build, floor)
        plan = _teacher_plan(record, build, floor)
    else:
        lines.append('Return one JSON line: {"action_index":N,"confidence":0.0,"action_scores":[{"action_index":N,"score":0.0}],"reason":"..."}')
        reason = f"{selected} matched the human card reward choice."
        plan = ""
    return [_row(
        system_prompt,
        "\n".join(lines),
        _assistant(picked_index, reason, plan=plan, action_count=len(choices) + 1),
        _meta(path, line_no, split, record, floor, "card_reward", selected),
    )]


def _relic_rows(
    path: Path,
    line_no: int,
    split: str,
    record: dict[str, Any],
    build: BuildState,
    floor: dict[str, Any],
    system_prompt: str,
    teacher_style: str,
) -> list[dict[str, Any]]:
    choices = [c for c in _as_list(floor.get("relic_choices")) if isinstance(c, dict)]
    if len(choices) <= 1:
        return []
    picked_index = next((i for i, choice in enumerate(choices) if choice.get("was_picked")), -1)
    if picked_index < 0:
        return []
    selected = _clean_id(choices[picked_index].get("relic_id"))
    lines = _base_lines(record, build, floor, "relic_select", include_strategy=teacher_style == "strategic_v2")
    lines.append("legal_actions:")
    for idx, choice in enumerate(choices):
        rid = _clean_id(choice.get("relic_id"))
        if teacher_style == "strategic_v2":
            lines.append(f"  [{idx}] pick_relic {_relic_line(rid, _display(choice, 'relic_id'))}")
        else:
            lines.append(f"  [{idx}] pick_relic {rid} | {_display(choice, 'relic_id')}")
    if teacher_style == "strategic_v2":
        lines.append('Return one JSON line: {"action_index":N,"confidence":0.0,"action_scores":[{"action_index":N,"score":0.0}],"plan":"...","reason":"..."}')
        reason = _selected_reason("relic_select", selected, record, build, floor)
        plan = _teacher_plan(record, build, floor)
    else:
        lines.append('Return one JSON line: {"action_index":N,"confidence":0.0,"action_scores":[{"action_index":N,"score":0.0}],"reason":"..."}')
        reason = f"{selected} matched the human relic choice."
        plan = ""
    return [_row(
        system_prompt,
        "\n".join(lines),
        _assistant(picked_index, reason, plan=plan, action_count=len(choices)),
        _meta(path, line_no, split, record, floor, "relic_select", selected),
    )]


def _rest_rows(
    path: Path,
    line_no: int,
    split: str,
    record: dict[str, Any],
    build: BuildState,
    floor: dict[str, Any],
    system_prompt: str,
    teacher_style: str,
) -> list[dict[str, Any]]:
    choice = str(floor.get("campfire_choice") or "").upper()
    if not choice:
        return []
    upgrades = [u for u in _as_list(floor.get("card_upgrades")) if isinstance(u, dict)]
    actions = ["SMITH", "REST"]
    picked_index = 0 if choice == "SMITH" else 1 if choice == "REST" else 0
    lines = _base_lines(record, build, floor, "rest_site_choice", include_strategy=teacher_style == "strategic_v2")
    lines.append("legal_actions:")
    upgrade_label = _clean_id(upgrades[0].get("card_id")) if upgrades else "best_card"
    if teacher_style == "strategic_v2" and upgrades:
        lines.append(_candidate_card_line(0, "smith", upgrade_label, _display(upgrades[0], "card_id")))
    else:
        lines.append(f"  [0] smith {upgrade_label}")
    lines.append("  [1] rest")
    selected = f"{actions[picked_index]} {upgrade_label if picked_index == 0 else ''}".strip()
    if teacher_style == "strategic_v2":
        lines.append('Return one JSON line: {"action_index":N,"confidence":0.0,"action_scores":[{"action_index":N,"score":0.0}],"plan":"...","reason":"..."}')
        reason = _selected_reason("rest_site_choice", selected, record, build, floor)
        plan = _teacher_plan(record, build, floor)
    else:
        lines.append('Return one JSON line: {"action_index":N,"confidence":0.0,"action_scores":[{"action_index":N,"score":0.0}],"reason":"..."}')
        reason = f"{selected} matched the human campfire choice."
        plan = ""
    return [_row(
        system_prompt,
        "\n".join(lines),
        _assistant(picked_index, reason, plan=plan, action_count=2),
        _meta(path, line_no, split, record, floor, "rest_site_choice", selected),
    )]


def _shop_rows(
    path: Path,
    line_no: int,
    split: str,
    record: dict[str, Any],
    build: BuildState,
    floor: dict[str, Any],
    system_prompt: str,
    teacher_style: str,
) -> list[dict[str, Any]]:
    actual_actions = [a for a in _as_list(floor.get("shop_actions")) if isinstance(a, dict)]
    if not actual_actions:
        return []
    legal: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(action_type: str, item_id: str) -> None:
        key = (action_type, item_id)
        if item_id and key not in seen:
            seen.add(key)
            legal.append(key)

    for choice in _as_list(floor.get("card_choices")):
        if isinstance(choice, dict):
            add("buy_card", _clean_id(choice.get("card_id")))
    for choice in _as_list(floor.get("relic_choices")):
        if isinstance(choice, dict):
            add("buy_relic", _clean_id(choice.get("relic_id")))
    for action in actual_actions:
        add(str(action.get("action_type") or "").lower(), _clean_id(action.get("item_id")))
    rows: list[dict[str, Any]] = []
    for action in actual_actions:
        target = (str(action.get("action_type") or "").lower(), _clean_id(action.get("item_id")))
        if target not in legal:
            continue
        picked_index = legal.index(target)
        selected = f"{target[0]} {target[1]}"
        lines = _base_lines(record, build, floor, "shop_choice", include_strategy=teacher_style == "strategic_v2")
        lines.append("legal_actions:")
        for idx, (action_type, item_id) in enumerate(legal):
            if teacher_style == "strategic_v2" and action_type == "buy_card":
                lines.append(_candidate_card_line(idx, action_type, item_id, item_id))
            elif teacher_style == "strategic_v2" and action_type == "buy_relic":
                lines.append(f"  [{idx}] buy_relic {_relic_line(item_id, include_description=False)}")
            else:
                lines.append(f"  [{idx}] {action_type} {item_id}")
        lines.append(f"  [{len(legal)}] leave_shop")
        if teacher_style == "strategic_v2":
            lines.append('Return one JSON line: {"action_index":N,"confidence":0.0,"action_scores":[{"action_index":N,"score":0.0}],"plan":"...","reason":"..."}')
            reason = _selected_reason("shop_choice", selected, record, build, floor)
            plan = _teacher_plan(record, build, floor)
        else:
            lines.append('Return one JSON line: {"action_index":N,"confidence":0.0,"action_scores":[{"action_index":N,"score":0.0}],"reason":"..."}')
            reason = f"{selected} matched the human shop action."
            plan = ""
        rows.append(_row(
            system_prompt,
            "\n".join(lines),
            _assistant(picked_index, reason, plan=plan, action_count=len(legal) + 1),
            _meta(path, line_no, split, record, floor, "shop_choice", selected),
        ))
    return rows


def _map_rows(path: Path, line_no: int, split: str, record: dict[str, Any], system_prompt: str, teacher_style: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run = record.get("run") or {}
    for act in _as_list(record.get("map_acts")):
        if not isinstance(act, dict):
            continue
        nodes = {
            tuple(node.get("coord") or []): node
            for node in _as_list(act.get("nodes"))
            if isinstance(node, dict) and isinstance(node.get("coord"), list)
        }
        visited = [tuple(coord) for coord in _as_list(act.get("visited_coords")) if isinstance(coord, list)]
        for index in range(max(0, len(visited) - 1)):
            current = visited[index]
            chosen = visited[index + 1]
            current_node = nodes.get(current) or {}
            children = [tuple(child) for child in _as_list(current_node.get("children")) if isinstance(child, list)]
            if len(children) <= 1 or chosen not in children:
                continue
            picked_index = children.index(chosen)
            lines = [
                "decision_type: map_choice",
                (
                    f"run: character={run.get('character')} ascension={run.get('ascension')} "
                    f"version={run.get('game_version')} victory={bool(run.get('is_victory'))}"
                ),
                f"map: act={act.get('act')} current={list(current)} boss={act.get('boss')}",
                "legal_actions:",
            ]
            for action_index, coord in enumerate(children):
                node = nodes.get(coord) or {}
                lines.append(f"  [{action_index}] choose_map_node coord={list(coord)} room={_room_name(node.get('type'))}")
            selected = f"coord={list(chosen)} room={_room_name((nodes.get(chosen) or {}).get('type'))}"
            if teacher_style == "strategic_v2":
                lines.insert(3, "current_plan:")
                lines.insert(4, "  core: choose a route that balances upgrades, shops, elite rewards, and HP safety")
                lines.insert(5, "  route_policy: prefer high-value paths when HP is safe; prefer rest/shop access when resources are low")
                lines.append('Return one JSON line: {"action_index":N,"confidence":0.0,"action_scores":[{"action_index":N,"score":0.0}],"plan":"...","reason":"..."}')
                reason = _selected_reason("map_choice", selected, record, BuildState.starter(str(run.get("character") or "")), {"floor": index + 1})
                plan = "Route plan: balance path rewards against HP safety and recovery access."
            else:
                lines.append('Return one JSON line: {"action_index":N,"confidence":0.0,"action_scores":[{"action_index":N,"score":0.0}],"reason":"..."}')
                reason = f"{selected} matched the human map route."
                plan = ""
            rows.append(_row(
                system_prompt,
                "\n".join(lines),
                _assistant(picked_index, reason, plan=plan, action_count=len(children)),
                _meta(path, line_no, split, record, {"floor": index + 1}, "map_choice", selected),
            ))
    return rows


def _meta(path: Path, line_no: int, split: str, record: dict[str, Any], floor: dict[str, Any], decision_type: str, selected: str) -> dict[str, Any]:
    run = record.get("run") or {}
    return {
        "source": "skada",
        "source_path": str(path),
        "source_line": line_no,
        "split": split,
        "decision_type": decision_type,
        "selected": selected,
        "run_id": run.get("run_id"),
        "character": run.get("character"),
        "ascension": run.get("ascension"),
        "game_version": run.get("game_version"),
        "is_victory": bool(run.get("is_victory")),
        "floor": floor.get("floor"),
    }


def _rows_from_record(path: Path, line_no: int, split: str, record: dict[str, Any], system_prompt: str, teacher_style: str) -> list[dict[str, Any]]:
    run = record.get("run") or {}
    build = BuildState.starter(str(run.get("character") or ""))
    rows: list[dict[str, Any]] = []
    rows.extend(_map_rows(path, line_no, split, record, system_prompt, teacher_style))
    for floor in sorted(_as_list(record.get("floor_timeline")), key=lambda item: int((item or {}).get("floor") or 0)):
        if not isinstance(floor, dict):
            continue
        build.update_resource_snapshot(floor)
        rows.extend(_card_reward_rows(path, line_no, split, record, build, floor, system_prompt, teacher_style))
        rows.extend(_relic_rows(path, line_no, split, record, build, floor, system_prompt, teacher_style))
        rows.extend(_rest_rows(path, line_no, split, record, build, floor, system_prompt, teacher_style))
        rows.extend(_shop_rows(path, line_no, split, record, build, floor, system_prompt, teacher_style))
        build.apply_floor_result(floor)
    return rows


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    type_cap_overrides = _parse_type_cap_overrides(args.max_per_type_overrides)
    out_dir = args.out_dir or (_STS2AI_ROOT / "Artifacts" / "llm" / "datasets" / f"skada_non_combat_{time.strftime('%Y%m%d-%H%M%S')}")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = load_system_prompt("non_combat")
    buckets: dict[str, list[dict[str, Any]]] = {}
    seen_counts: Counter[str] = Counter()
    processed_runs = 0

    for path, line_no, split, record in _iter_records(args.details_root.resolve(), include_failure=args.include_failure):
        run = record.get("run") or {}
        if args.character and str(run.get("character") or "").upper() != args.character.upper():
            continue
        if args.game_version and str(run.get("game_version") or "") != args.game_version:
            continue
        processed_runs += 1
        for row in _rows_from_record(path, line_no, split, record, system_prompt, args.teacher_style):
            decision_type = str((row.get("meta") or {}).get("decision_type") or "")
            bucket = buckets.setdefault(decision_type, [])
            seen_counts[decision_type] += 1
            cap = type_cap_overrides.get(decision_type, args.max_per_type)
            if cap <= 0 or len(bucket) < cap:
                bucket.append(row)
                continue
            replacement_index = rng.randrange(seen_counts[decision_type])
            if replacement_index < cap:
                bucket[replacement_index] = row
        if args.max_runs > 0 and processed_runs >= args.max_runs:
            break

    rows: list[dict[str, Any]] = [row for bucket in buckets.values() for row in bucket]
    rng.shuffle(rows)
    eval_size = int(len(rows) * max(0.0, min(0.5, args.eval_ratio)))
    eval_rows = rows[:eval_size]
    train_rows = rows[eval_size:]

    def dump_jsonl(path: Path, payloads: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    dump_jsonl(out_dir / "train.jsonl", train_rows)
    dump_jsonl(out_dir / "eval.jsonl", eval_rows)
    summary = {
        "kind": "skada_non_combat_sft_dataset",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "details_root": str(args.details_root.resolve()),
        "out_dir": str(out_dir),
        "processed_runs": processed_runs,
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "decision_counts": {key: len(value) for key, value in buckets.items()},
        "decision_seen_counts": dict(seen_counts),
        "skipped": {
            f"{key}_reservoir": max(0, seen_counts[key] - len(value))
            for key, value in buckets.items()
            if seen_counts[key] > len(value)
        },
        "args": vars(args) | {"out_dir": str(out_dir), "details_root": str(args.details_root)},
        "outputs": {
            "train": str(out_dir / "train.jsonl"),
            "eval": str(out_dir / "eval.jsonl"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
