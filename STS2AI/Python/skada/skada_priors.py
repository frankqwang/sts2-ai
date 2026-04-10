"""Skada community priors for STS2AI training integration.

Loads Skada analytics DB and provides fast lookup dicts for:
- Card quality priors (skada_score, pick_rate, win_rate_delta, floor-value)
- Relic quality priors (pick_rate, win_rate_owned, hold_rate)
- Card synergy map (synergy_lift between card pairs)
- Boss difficulty (wipe_rate, best_cards)

Usage:
    from skada.skada_priors import SkadaPriors
    priors = SkadaPriors()  # auto-detects DB path
    card = priors.card("bloodletting")  # CardPrior namedtuple
    relic = priors.relic("burning_blood")
    syn = priors.synergy("bloodletting", "offering")
    boss = priors.boss("THE_KIN_BOSS")
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path(__file__).resolve().parents[2] / "Assets" / "datasets" / "skada" / "skada_analytics.sqlite"


@dataclass(frozen=True, slots=True)
class CardPrior:
    skada_score_norm: float  # [0, 1] normalized within character
    pick_rate: float
    win_rate_delta: float    # win_rate_picked - win_rate_skipped
    hold_rate: float
    floor_early: float       # pick_rate * win_delta in early game
    floor_mid: float
    floor_late: float
    character: str


@dataclass(frozen=True, slots=True)
class RelicPrior:
    pick_rate: float
    win_rate_owned: float
    hold_rate: float
    win_rate_delta: float    # win_rate_picked - win_rate_skipped


@dataclass(frozen=True, slots=True)
class BossDifficulty:
    wipe_rate: float
    win_avg_dpt: float
    lose_avg_dpt: float
    best_cards: list[str]    # top card_ids (lowercase slugs)


class SkadaPriors:
    """Fast lookup interface for Skada community priors."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB
        self._cards: dict[str, CardPrior] = {}
        self._relics: dict[str, RelicPrior] = {}
        self._synergies: dict[tuple[str, str], float] = {}
        self._bosses: dict[str, BossDifficulty] = {}
        self._loaded = False

        if self.db_path.exists():
            self._load()
        else:
            logger.warning("Skada DB not found at %s", self.db_path)

    @property
    def loaded(self) -> bool:
        return self._loaded

    def card(self, slug: str) -> CardPrior | None:
        """Look up card prior by lowercase slug (e.g. 'bloodletting')."""
        return self._cards.get(slug)

    def relic(self, slug: str) -> RelicPrior | None:
        """Look up relic prior by lowercase slug (e.g. 'burning_blood')."""
        return self._relics.get(slug)

    def synergy(self, card_a: str, card_b: str) -> float:
        """Get synergy lift between two cards (lowercase slugs). Returns 0.0 if unknown."""
        key = (min(card_a, card_b), max(card_a, card_b))
        return self._synergies.get(key, 0.0)

    def deck_synergy_boost(self, card_slug: str, deck_slugs: list[str]) -> float:
        """Compute total synergy boost of adding a card to a deck.
        Returns sum of synergy_lift for all deck cards that have known synergy."""
        total = 0.0
        for d in deck_slugs:
            total += self.synergy(card_slug, d)
        return total

    def boss(self, encounter: str) -> BossDifficulty | None:
        """Look up boss difficulty by encounter ID (e.g. 'THE_KIN_BOSS')."""
        return self._bosses.get(encounter)

    def card_score_for_context(
        self,
        card_slug: str,
        floor: int,
        deck_slugs: list[str] | None = None,
    ) -> float:
        """Compute context-aware card quality score [0, 1].

        Blends base skada_score with floor-conditional value and deck synergy.
        """
        cp = self._cards.get(card_slug)
        if cp is None:
            return 0.5  # neutral prior for unknown cards

        # Base score
        score = cp.skada_score_norm

        # Floor-conditional adjustment
        if floor <= 6:
            floor_val = cp.floor_early
        elif floor <= 12:
            floor_val = cp.floor_mid
        else:
            floor_val = cp.floor_late

        if floor_val != 0.0:
            score = 0.7 * score + 0.3 * floor_val

        # Synergy boost (clamped)
        if deck_slugs:
            syn_boost = self.deck_synergy_boost(card_slug, deck_slugs)
            score += min(0.15, syn_boost * 0.05)

        return max(0.0, min(1.0, score))

    @property
    def num_cards(self) -> int:
        return len(self._cards)

    @property
    def num_relics(self) -> int:
        return len(self._relics)

    @property
    def num_synergies(self) -> int:
        return len(self._synergies)

    @property
    def num_bosses(self) -> int:
        return len(self._bosses)

    def _load(self) -> None:
        """Load all priors from SQLite DB."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            self._load_cards(conn)
            self._load_relics(conn)
            self._load_synergies(conn)
            self._load_bosses(conn)
            conn.close()
            self._loaded = True
            logger.info(
                "Skada priors loaded: %d cards, %d relics, %d synergy pairs, %d bosses",
                len(self._cards), len(self._relics),
                len(self._synergies), len(self._bosses),
            )
        except Exception as e:
            logger.error("Failed to load Skada priors: %s", e)

    def _load_cards(self, conn: sqlite3.Connection) -> None:
        """Load card priors with per-character normalization."""
        rows = conn.execute("""
            SELECT card_id, character, skada_score, pick_rate,
                   win_rate_picked, win_rate_skipped, hold_rate
            FROM cards WHERE skada_score IS NOT NULL
        """).fetchall()

        # Compute per-character min/max for normalization
        char_ranges: dict[str, tuple[float, float]] = {}
        for r in rows:
            ch = r["character"] or "UNKNOWN"
            s = r["skada_score"] or 0.0
            lo, hi = char_ranges.get(ch, (s, s))
            char_ranges[ch] = (min(lo, s), max(hi, s))

        # Load floor-conditional values
        floor_data: dict[str, dict[str, float]] = {}
        try:
            fv_rows = conn.execute("""
                SELECT card_id, stage, pick_rate, win_rate_picked, win_rate_skipped
                FROM card_floor_value
            """).fetchall()
            for fv in fv_rows:
                card_id = fv["card_id"]
                stage = fv["stage"]
                # Use pick_rate * win_rate_delta as floor value signal
                pr = fv["pick_rate"] or 0.0
                wr_p = fv["win_rate_picked"] or 0.0
                wr_s = fv["win_rate_skipped"] or 0.0
                delta = wr_p - wr_s
                value = max(0.0, min(1.0, 0.5 + delta * 2.0))  # center at 0.5
                floor_data.setdefault(card_id, {})[stage] = value
        except Exception:
            pass

        for r in rows:
            card_id = r["card_id"]
            slug = card_id.lower()
            ch = r["character"] or "UNKNOWN"
            s = r["skada_score"] or 0.0

            lo, hi = char_ranges.get(ch, (0.0, 1.0))
            rng = hi - lo if hi > lo else 1.0
            norm = (s - lo) / rng

            wr_p = r["win_rate_picked"] or 0.0
            wr_s = r["win_rate_skipped"] or 0.0
            delta = wr_p - wr_s

            fv = floor_data.get(card_id, {})

            self._cards[slug] = CardPrior(
                skada_score_norm=round(norm, 4),
                pick_rate=r["pick_rate"] or 0.0,
                win_rate_delta=round(delta, 4),
                hold_rate=r["hold_rate"] or 0.0,
                floor_early=fv.get("early", 0.5),
                floor_mid=fv.get("mid", 0.5),
                floor_late=fv.get("late", 0.5),
                character=ch,
            )

    def _load_relics(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute("""
            SELECT relic_id, pick_rate, win_rate_owned, hold_rate,
                   win_rate_picked, win_rate_skipped
            FROM relics
        """).fetchall()

        for r in rows:
            slug = r["relic_id"].lower()
            wr_p = r["win_rate_picked"] or 0.0
            wr_s = r["win_rate_skipped"] or 0.0
            self._relics[slug] = RelicPrior(
                pick_rate=r["pick_rate"] or 0.0,
                win_rate_owned=r["win_rate_owned"] or 0.0,
                hold_rate=r["hold_rate"] or 0.0,
                win_rate_delta=round(wr_p - wr_s, 4),
            )

    def _load_synergies(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute("""
            SELECT card_a, card_b, synergy_lift
            FROM card_companions
            WHERE synergy_lift IS NOT NULL
        """).fetchall()

        for r in rows:
            a = r["card_a"].lower()
            b = r["card_b"].lower()
            key = (min(a, b), max(a, b))
            self._synergies[key] = r["synergy_lift"]

    def _load_bosses(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute("""
            SELECT encounter, wipe_rate, win_avg_dpt, lose_avg_dpt
            FROM boss_guide
        """).fetchall()

        for r in rows:
            enc = r["encounter"]
            best = conn.execute("""
                SELECT card_id FROM boss_best_cards
                WHERE encounter = ? ORDER BY dmg_per_play DESC LIMIT 5
            """, (enc,)).fetchall()
            best_slugs = [c["card_id"].lower() for c in best]

            self._bosses[enc] = BossDifficulty(
                wipe_rate=r["wipe_rate"] or 0.0,
                win_avg_dpt=r["win_avg_dpt"] or 0.0,
                lose_avg_dpt=r["lose_avg_dpt"] or 0.0,
                best_cards=best_slugs,
            )
