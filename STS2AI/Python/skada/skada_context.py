#!/usr/bin/env python3
"""Shared context helpers for Skada-derived supervised datasets."""

from __future__ import annotations

import sqlite3
from collections import Counter
from typing import Any

from skada_priors import SkadaPriors


CURSE_SLUGS = {
    "ascenders_bane",
    "burn",
    "clumsy",
    "curse_of_the_bell",
    "decay",
    "debt",
    "doubt",
    "dazed",
    "guilty",
    "injury",
    "latern_key",
    "lantern_key",
    "normality",
    "pain",
    "poor_sleep",
    "regret",
    "shame",
    "spoils_map",
    "wound",
}

STARTER_DECKS: dict[str, list[tuple[str, int]]] = {
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
    "REGENT": [
        ("STRIKE_REGENT", 5),
        ("DEFEND_REGENT", 4),
        ("FALLING_STAR", 1),
        ("VENERATE", 1),
        ("GLOW", 1),
    ],
    "NECROBINDER": [
        ("STRIKE_NECROBINDER", 5),
        ("DEFEND_NECROBINDER", 4),
        ("BODYGUARD", 1),
        ("INVOKE", 1),
    ],
}


def slugify(card_id: str | None) -> str:
    text = str(card_id or "").strip().lower()
    if text.endswith("+"):
        text = text[:-1]
    return text


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def dict_factory(cursor: sqlite3.Cursor, row: sqlite3.Row) -> dict[str, Any]:
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def fetch_many(conn: sqlite3.Connection, query: str, run_ids: list[int]) -> list[dict[str, Any]]:
    if not run_ids:
        return []
    result: list[dict[str, Any]] = []
    chunk_size = 500
    for start in range(0, len(run_ids), chunk_size):
        chunk = run_ids[start:start + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        result.extend(conn.execute(query.format(placeholders=placeholders), chunk).fetchall())
    return result


def act_from_floor(floor: int) -> int:
    if floor <= 17:
        return 1
    if floor <= 33:
        return 2
    return 3


def is_basic_slug(slug: str) -> bool:
    return slug.startswith("strike_") or slug.startswith("defend_")


def is_curse_slug(slug: str) -> bool:
    return slug in CURSE_SLUGS


class DeckTracker:
    """Approximate run-state tracker for non-combat supervised samples."""

    def __init__(self, character: str):
        self.character = str(character or "UNKNOWN").upper()
        self.deck_counts: Counter[str] = Counter()
        self.upgrade_counts: Counter[str] = Counter()
        self.relic_ids: list[str] = ["starter_relic"]
        self.history: list[str] = []
        self.card_gains = 0
        self.shop_removes = 0
        self.upgrade_events = 0
        self.canonical_ids: dict[str, str] = {}

        for card_id, count in STARTER_DECKS.get(self.character, []):
            slug = slugify(card_id)
            self.deck_counts[slug] += int(count)
            self.canonical_ids[slug] = card_id

    def _remember_card(self, card_id: str | None) -> str:
        raw = str(card_id or "").strip()
        slug = slugify(raw)
        if slug and raw:
            self.canonical_ids[slug] = raw.rstrip("+")
        return slug

    def add_card(self, card_id: str | None) -> None:
        slug = self._remember_card(card_id)
        if not slug:
            return
        self.deck_counts[slug] += 1
        self.history.append(slug)
        self.card_gains += 1

    def remove_card(self, card_id: str | None) -> None:
        slug = self._remember_card(card_id)
        if not slug:
            return
        if self.deck_counts.get(slug, 0) > 0:
            self.deck_counts[slug] -= 1
            if self.upgrade_counts.get(slug, 0) > self.deck_counts[slug]:
                self.upgrade_counts[slug] = self.deck_counts[slug]
            if self.deck_counts[slug] <= 0:
                self.deck_counts.pop(slug, None)
                self.upgrade_counts.pop(slug, None)
        self.shop_removes += 1

    def upgrade_card(self, card_id: str | None) -> None:
        slug = self._remember_card(card_id)
        if not slug:
            return
        current_count = max(1, self.deck_counts.get(slug, 0))
        self.deck_counts[slug] = current_count
        self.upgrade_counts[slug] = min(current_count, self.upgrade_counts.get(slug, 0) + 1)
        self.upgrade_events += 1

    def add_relic(self, relic_id: str | None) -> None:
        rid = str(relic_id or "").strip().lower()
        if rid:
            self.relic_ids.append(rid)

    def current_history(self, limit: int = 40) -> list[str]:
        return self.history[-limit:]

    def starter_slugs(self) -> set[str]:
        return {slugify(card_id) for card_id, _ in STARTER_DECKS.get(self.character, [])}

    def deck_size(self) -> int:
        return sum(int(count) for count in self.deck_counts.values())

    def deck_summary(self) -> dict[str, Any]:
        top_cards = sorted(
            (
                {
                    "card_id": self.canonical_ids.get(slug, slug.upper()),
                    "count": int(count),
                    "upgraded": int(self.upgrade_counts.get(slug, 0)),
                }
                for slug, count in self.deck_counts.items()
                if count > 0
            ),
            key=lambda item: (-item["count"], item["card_id"]),
        )[:8]
        starter_slugs = self.starter_slugs()
        return {
            "deck_size": self.deck_size(),
            "unique_cards": len([1 for count in self.deck_counts.values() if count > 0]),
            "starter_cards_remaining": int(sum(self.deck_counts.get(slug, 0) for slug in starter_slugs)),
            "recent_cards": [self.canonical_ids.get(slug, slug.upper()) for slug in self.current_history(8)],
            "top_cards": top_cards,
        }

    def upgradeable_options(self, priors: SkadaPriors, floor: int) -> list[dict[str, Any]]:
        history = self.current_history()
        options: list[dict[str, Any]] = []
        for slug, count in sorted(self.deck_counts.items()):
            unupgraded = int(count) - int(self.upgrade_counts.get(slug, 0))
            if unupgraded <= 0:
                continue
            options.append({
                "card_id": self.canonical_ids.get(slug, slug.upper()),
                "count_in_deck": int(count),
                "unupgraded_count": int(unupgraded),
                "context_score": priors.card_score_for_context(slug, floor, history),
                "is_basic": 1 if is_basic_slug(slug) else 0,
                "is_curse": 1 if is_curse_slug(slug) else 0,
            })
        options.sort(key=lambda item: (-item["context_score"], item["card_id"]))
        return options

    def remove_candidates(self, priors: SkadaPriors, floor: int, chosen_card_id: str | None = None) -> list[dict[str, Any]]:
        history = self.current_history()
        counts = Counter(self.deck_counts)
        chosen_slug = slugify(chosen_card_id)
        if chosen_slug and counts.get(chosen_slug, 0) <= 0:
            counts[chosen_slug] = 1
            if chosen_card_id:
                self.canonical_ids[chosen_slug] = str(chosen_card_id).rstrip("+")

        starter_slugs = self.starter_slugs()
        options: list[dict[str, Any]] = []
        for slug, count in sorted(counts.items()):
            context_score = priors.card_score_for_context(slug, floor, history)
            is_basic = 1 if is_basic_slug(slug) else 0
            is_curse = 1 if is_curse_slug(slug) else 0
            is_starter_unique = 1 if slug in starter_slugs and not is_basic else 0
            remove_score = (
                (1.0 - context_score)
                + 0.25 * is_basic
                + 0.20 * is_curse
                + 0.10 * is_starter_unique
                + 0.03 * min(4.0, float(count))
            )
            options.append({
                "item_id": self.canonical_ids.get(slug, slug.upper()),
                "count_in_deck": int(count),
                "is_basic": is_basic,
                "is_curse": is_curse,
                "is_starter_unique": is_starter_unique,
                "context_score": context_score,
                "baseline_score": remove_score,
            })
        options.sort(key=lambda item: item["item_id"])
        return options

    def shared_context(
        self,
        priors: SkadaPriors,
        floor: int,
        hp_before: float,
        gold_before: float,
        recent_damage_taken: float,
        recent_damage_dealt: float,
        recent_shop_visits: int,
        recent_elites: int,
    ) -> dict[str, Any]:
        history = self.current_history()
        upgradeable = self.upgradeable_options(priors, floor)
        avg_history_score = (
            sum(priors.card_score_for_context(card_slug, floor, history) for card_slug in history) / len(history)
            if history else 0.5
        )
        best_upgrade_score = max((item["context_score"] for item in upgradeable), default=0.5)
        mean_upgrade_score = (
            sum(item["context_score"] for item in upgradeable) / len(upgradeable)
            if upgradeable else 0.5
        )
        basic_card_count = int(sum(count for slug, count in self.deck_counts.items() if is_basic_slug(slug)))
        curse_card_count = int(sum(count for slug, count in self.deck_counts.items() if is_curse_slug(slug)))
        return {
            "act": act_from_floor(floor),
            "hp_before": safe_float(hp_before),
            "gold_before": safe_float(gold_before),
            "prior_card_count": self.deck_size(),
            "prior_relic_count": max(1, len(self.relic_ids)),
            "prior_card_gains": int(self.card_gains),
            "prior_shop_removes": int(self.shop_removes),
            "prior_upgrades": int(self.upgrade_events),
            "deck_size": self.deck_size(),
            "unique_cards": len([1 for count in self.deck_counts.values() if count > 0]),
            "upgraded_cards": int(sum(self.upgrade_counts.values())),
            "upgradeable_count": len(upgradeable),
            "basic_card_count": basic_card_count,
            "curse_card_count": curse_card_count,
            "avg_history_context_score": round(avg_history_score, 6),
            "best_upgrade_context_score": round(best_upgrade_score, 6),
            "mean_upgrade_context_score": round(mean_upgrade_score, 6),
            "recent_damage_taken": round(safe_float(recent_damage_taken), 4),
            "recent_damage_dealt": round(safe_float(recent_damage_dealt), 4),
            "recent_shop_visits": int(recent_shop_visits),
            "recent_elites": int(recent_elites),
            "deck_summary": self.deck_summary(),
            "upgradeable_options": upgradeable[:20],
        }
