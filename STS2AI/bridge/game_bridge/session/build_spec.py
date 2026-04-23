"""session 共用的 build 规格归一。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CardSpecPy:
    id: str
    upgrade_level: int = 0
    floor_added_to_deck: int | None = None
    props: dict[str, Any] | None = None

    @classmethod
    def from_value(cls, value: Any) -> "CardSpecPy":
        if isinstance(value, str):
            card_id = value.strip()
            if not card_id:
                raise ValueError("build.deck contains an empty card id")
            return cls(id=card_id)
        if not isinstance(value, dict):
            raise TypeError("build.deck entries must be strings or dicts")
        card_id = str(value.get("id") or value.get("card_id") or value.get("name") or "").strip()
        if not card_id:
            raise ValueError("build.deck entry is missing id")
        upgrade_level = value.get("upgrade_level", value.get("upgrades", value.get("current_upgrade_level")))
        if upgrade_level is None and bool(value.get("is_upgraded")):
            upgrade_level = 1
        props = value.get("props")
        if props is not None and not isinstance(props, dict):
            raise TypeError("build.deck entry props must be a dict")
        floor_added = value.get("floor_added_to_deck")
        return cls(
            id=card_id,
            upgrade_level=max(0, int(upgrade_level or 0)),
            floor_added_to_deck=None if floor_added is None else int(floor_added),
            props=props,
        )

    def to_sim_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": str(self.id),
            "upgrade_level": int(self.upgrade_level),
        }
        if self.floor_added_to_deck is not None:
            payload["floor_added_to_deck"] = int(self.floor_added_to_deck)
        if self.props:
            payload["props"] = dict(self.props)
        return payload


@dataclass(slots=True)
class RelicSpecPy:
    id: str
    floor_added_to_deck: int | None = None

    @classmethod
    def from_value(cls, value: Any) -> "RelicSpecPy":
        if isinstance(value, str):
            relic_id = value.strip()
            if not relic_id:
                raise ValueError("build.relics contains an empty relic id")
            return cls(id=relic_id)
        if not isinstance(value, dict):
            raise TypeError("build.relics entries must be strings or dicts")
        relic_id = str(value.get("id") or value.get("relic_id") or value.get("name") or "").strip()
        if not relic_id:
            raise ValueError("build.relics entry is missing id")
        floor_added = value.get("floor_added_to_deck")
        return cls(
            id=relic_id,
            floor_added_to_deck=None if floor_added is None else int(floor_added),
        )

    def to_sim_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"id": str(self.id)}
        if self.floor_added_to_deck is not None:
            payload["floor_added_to_deck"] = int(self.floor_added_to_deck)
        return payload


@dataclass(slots=True)
class PotionSpecPy:
    id: str
    slot: int | None = None

    @classmethod
    def from_value(cls, value: Any) -> "PotionSpecPy":
        if isinstance(value, str):
            potion_id = value.strip()
            if not potion_id:
                raise ValueError("build.potions contains an empty potion id")
            return cls(id=potion_id)
        if not isinstance(value, dict):
            raise TypeError("build.potions entries must be strings or dicts")
        potion_id = str(value.get("id") or value.get("potion_id") or value.get("name") or "").strip()
        if not potion_id:
            raise ValueError("build.potions entry is missing id")
        slot = value.get("slot", value.get("slot_index"))
        return cls(
            id=potion_id,
            slot=None if slot is None else max(0, int(slot)),
        )

    def to_sim_dict(self, fallback_slot: int) -> dict[str, Any]:
        slot = fallback_slot if self.slot is None else max(0, int(self.slot))
        return {
            "id": str(self.id),
            "slot": slot,
        }


def _extract_optional_int(value: dict[str, Any], *aliases: str) -> int | None:
    for alias in aliases:
        if alias in value and value[alias] is not None:
            return int(value[alias])
    return None


@dataclass(slots=True)
class BuildSpecPy:
    deck: list[CardSpecPy] = field(default_factory=list)
    relics: list[RelicSpecPy] = field(default_factory=list)
    potions: list[PotionSpecPy] = field(default_factory=list)
    current_hp: int | None = None
    max_hp: int | None = None
    max_energy: int | None = None
    max_potion_slots: int | None = None
    gold: int | None = None
    floor: int | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BuildSpecPy":
        if not isinstance(value, dict):
            raise TypeError("build must be a dict when provided")

        raw_value = value
        nested_build = raw_value.get("build")
        if nested_build is not None:
            if not isinstance(nested_build, dict):
                raise TypeError("build.build must be a dict when provided")
            value = nested_build

        deck_entries = value.get("deck", value.get("cards")) or []
        if not isinstance(deck_entries, list):
            raise TypeError("build.deck must be a list")

        relic_entries = value.get("relics", value.get("relic_ids")) or []
        if not isinstance(relic_entries, list):
            raise TypeError("build.relics must be a list")

        potion_entries = value.get("potions", value.get("potion_ids")) or []
        if not isinstance(potion_entries, list):
            raise TypeError("build.potions must be a list")

        return cls(
            deck=[CardSpecPy.from_value(entry) for entry in deck_entries],
            relics=[RelicSpecPy.from_value(entry) for entry in relic_entries],
            potions=[PotionSpecPy.from_value(entry) for entry in potion_entries],
            current_hp=_extract_optional_int(value, "current_hp", "hp"),
            max_hp=_extract_optional_int(value, "max_hp"),
            max_energy=_extract_optional_int(value, "max_energy", "energy"),
            max_potion_slots=_extract_optional_int(value, "max_potion_slots", "max_potions", "potion_slot_count"),
            gold=_extract_optional_int(value, "gold"),
            floor=(
                _extract_optional_int(raw_value, "floor", "current_floor", "run_floor")
                if value is not raw_value
                else None
            )
            or _extract_optional_int(value, "floor", "current_floor", "run_floor"),
        )

    def to_sim_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.deck:
            payload["deck"] = [card.to_sim_dict() for card in self.deck]
        if self.relics:
            payload["relics"] = [relic.to_sim_dict() for relic in self.relics]
        if self.potions:
            seen_slots: set[int] = set()
            potions_payload: list[dict[str, Any]] = []
            for fallback_slot, potion in enumerate(self.potions):
                potion_payload = potion.to_sim_dict(fallback_slot=fallback_slot)
                slot = potion_payload["slot"]
                if slot in seen_slots:
                    raise ValueError(f"build.potions contains duplicate slot {slot}")
                seen_slots.add(slot)
                potions_payload.append(potion_payload)
            payload["potions"] = potions_payload
        for key in ("current_hp", "max_hp", "max_energy", "max_potion_slots", "gold", "floor"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = int(value)
        return payload


def normalize_build_spec(build: BuildSpecPy | dict[str, Any] | None) -> dict[str, Any] | None:
    if build is None:
        return None
    if isinstance(build, BuildSpecPy):
        normalized = build.to_sim_dict()
        return normalized or None
    if isinstance(build, dict):
        normalized = BuildSpecPy.from_dict(build).to_sim_dict()
        return normalized or None
    raise TypeError("build must be a dict, BuildSpecPy, or None")


__all__ = [
    "BuildSpecPy",
    "CardSpecPy",
    "PotionSpecPy",
    "RelicSpecPy",
    "normalize_build_spec",
]
