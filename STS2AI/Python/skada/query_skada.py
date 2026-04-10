#!/usr/bin/env python3
"""
Query interface for Skada Analytics data.

Provides card tier lists, synergy pairs, boss strategies, and other
community statistics for use by the STS2 AI training pipeline.

Usage as CLI:
    python STS2AI/Python/skada/query_skada.py card-tier IRONCLAD
    python STS2AI/Python/skada/query_skada.py synergies IRONCLAD
    python STS2AI/Python/skada/query_skada.py boss-guide THE_KIN_BOSS
    python STS2AI/Python/skada/query_skada.py card-info BLOODLETTING
    python STS2AI/Python/skada/query_skada.py best-picks IRONCLAD --floor early
    python STS2AI/Python/skada/query_skada.py relic-tier
    python STS2AI/Python/skada/query_skada.py encounter-danger
    python STS2AI/Python/skada/query_skada.py deck-size IRONCLAD
    python STS2AI/Python/skada/query_skada.py overview
    python STS2AI/Python/skada/query_skada.py export-card-priors IRONCLAD

Usage as module:
    from skada.query_skada import SkadaQuery
    sq = SkadaQuery()
    tier = sq.card_tier_list("IRONCLAD", top_n=20)
"""
import argparse
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(__file__))
from skada_db import DB_PATH


class SkadaQuery:
    def __init__(self, db_path=None):
        self.db_path = str(db_path or DB_PATH)
        self._conn = None

    @property
    def conn(self):
        if self._conn is None:
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(
                    f"Skada DB not found at {self.db_path}. "
                    "Run scrape_skada.py first."
                )
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Card queries ──

    def card_tier_list(self, character=None, top_n=50, sort_by="skada_score"):
        """Get card tier list sorted by skada_score (or pick_rate, win_rate_delta)."""
        valid_sorts = {"skada_score", "pick_rate", "win_rate_picked",
                       "win_rate_delta", "delta_vs_skipped", "deck_win_rate"}
        if sort_by not in valid_sorts:
            sort_by = "skada_score"

        where = ""
        params = []
        if character:
            where = "WHERE character = ?"
            params.append(character)

        rows = self.conn.execute(f"""
            SELECT card_id, name_en, name_zh, character, rank, skada_score,
                   pick_rate, win_rate_picked, win_rate_skipped, win_rate_delta,
                   delta_vs_skipped, deck_win_rate, deck_runs, hold_rate, seen,
                   dmg_per_play, blk_per_play, plays_per_combat
            FROM cards {where}
            ORDER BY {sort_by} DESC
            LIMIT ?
        """, params + [top_n]).fetchall()
        return [dict(r) for r in rows]

    def card_info(self, card_id):
        """Get detailed info for a specific card."""
        card = self.conn.execute(
            "SELECT * FROM cards WHERE card_id = ?", (card_id,)
        ).fetchone()
        if not card:
            return None
        result = dict(card)
        result["obtain_sources"] = [dict(r) for r in self.conn.execute(
            "SELECT * FROM card_obtain_sources WHERE card_id = ?", (card_id,)
        ).fetchall()]
        result["floor_value"] = [dict(r) for r in self.conn.execute(
            "SELECT * FROM card_floor_value WHERE card_id = ?", (card_id,)
        ).fetchall()]
        return result

    def best_picks_by_floor(self, character=None, stage="early", top_n=20):
        """Get best card picks by game stage (early/mid/late)."""
        where_parts = ["stage = ?"]
        params = [stage]
        if character:
            where_parts.append("character = ?")
            params.append(character)
        where = " AND ".join(where_parts)

        return [dict(r) for r in self.conn.execute(f"""
            SELECT card_id, character, stage, pick_rate,
                   win_rate_picked, win_rate_skipped, delta_vs_community,
                   seen, picked, confidence
            FROM card_floor_value
            WHERE {where} AND confidence = 'high' AND seen >= 50
            ORDER BY delta_vs_community DESC
            LIMIT ?
        """, params + [top_n]).fetchall()]

    def card_synergies(self, character=None, top_n=20):
        """Get top card synergy pairs."""
        where = ""
        params = []
        if character:
            where = "WHERE character = ?"
            params.append(character)

        return [dict(r) for r in self.conn.execute(f"""
            SELECT card_a, card_b, character, pair_runs, pair_win_rate,
                   card_a_solo_wr, card_b_solo_wr, synergy_lift, confidence
            FROM card_companions {where}
            ORDER BY synergy_lift DESC
            LIMIT ?
        """, params + [top_n]).fetchall()]

    # ── Relic queries ──

    def relic_tier_list(self, top_n=30, sort_by="win_rate_picked"):
        """Get relic tier list."""
        valid_sorts = {"win_rate_picked", "pick_rate", "win_rate_owned", "hold_rate"}
        if sort_by not in valid_sorts:
            sort_by = "win_rate_picked"
        return [dict(r) for r in self.conn.execute(f"""
            SELECT relic_id, name_en, name_zh, seen, pick_rate, skip_rate,
                   win_rate_picked, win_rate_skipped, win_rate_owned, hold_rate
            FROM relics
            ORDER BY {sort_by} DESC
            LIMIT ?
        """, (top_n,)).fetchall()]

    # ── Encounter queries ──

    def encounter_danger_list(self, enc_type=None, top_n=30):
        """Get encounters sorted by wipe rate (most dangerous first)."""
        where = ""
        params = []
        if enc_type:
            where = "WHERE enc_type = ?"
            params.append(enc_type)
        return [dict(r) for r in self.conn.execute(f"""
            SELECT encounter, enc_type, type, name_en, name_zh,
                   times_seen, avg_turns, avg_damage_taken, avg_dpt, wipe_rate
            FROM encounters {where}
            ORDER BY wipe_rate DESC
            LIMIT ?
        """, params + [top_n]).fetchall()]

    # ── Boss queries ──

    def boss_guide(self, encounter=None):
        """Get boss guide with best cards."""
        if encounter:
            bosses = [dict(r) for r in self.conn.execute(
                "SELECT * FROM boss_guide WHERE encounter = ?", (encounter,)
            ).fetchall()]
        else:
            bosses = [dict(r) for r in self.conn.execute(
                "SELECT * FROM boss_guide ORDER BY wipe_rate DESC"
            ).fetchall()]

        for boss in bosses:
            boss["best_cards"] = [dict(r) for r in self.conn.execute(
                "SELECT * FROM boss_best_cards WHERE encounter = ? ORDER BY dmg_per_play DESC",
                (boss["encounter"],)
            ).fetchall()]
        return bosses

    # ── Deck size queries ──

    def optimal_deck_size(self, character):
        """Get deck size vs win rate curve for a character."""
        return [dict(r) for r in self.conn.execute("""
            SELECT deck_size, total, wins, win_rate
            FROM deck_size_curves
            WHERE character = ? AND total >= 20
            ORDER BY deck_size
        """, (character,)).fetchall()]

    # ── Run statistics ──

    def run_stats(self, character=None):
        """Get aggregate run statistics."""
        where = ""
        params = []
        if character:
            where = "WHERE character = ?"
            params.append(character)

        row = self.conn.execute(f"""
            SELECT COUNT(*) as total_runs,
                   SUM(is_victory) as wins,
                   ROUND(100.0 * SUM(is_victory) / COUNT(*), 2) as win_rate,
                   ROUND(AVG(floor_reached), 1) as avg_floor,
                   ROUND(AVG(duration_sec), 0) as avg_duration
            FROM runs {where}
        """, params).fetchone()
        return dict(row) if row else None

    def death_cause_stats(self, character=None, top_n=15):
        """Get most common death causes."""
        where = "WHERE death_cause IS NOT NULL"
        params = []
        if character:
            where += " AND character = ?"
            params.append(character)

        return [dict(r) for r in self.conn.execute(f"""
            SELECT death_cause, COUNT(*) as deaths,
                   ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM runs
                       {'WHERE character = ?' if character else ''}
                   ), 2) as pct
            FROM runs {where}
            GROUP BY death_cause
            ORDER BY deaths DESC
            LIMIT ?
        """, params + (params[:] if character else []) + [top_n]).fetchall()]

    # ── Overview ──

    def overview(self):
        """Get overall platform statistics."""
        stats = {}
        for row in self.conn.execute("SELECT key, value FROM overview"):
            stats[row["key"]] = row["value"]
        stats["character_win_rates"] = [dict(r) for r in self.conn.execute(
            "SELECT * FROM character_win_rates"
        ).fetchall()]
        return stats

    # ── Export for AI ──

    def export_card_priors(self, character):
        """
        Export card quality priors for AI training.
        Returns a dict mapping card_id -> normalized score [0, 1].
        Based on skada_score normalized to [0, 1] range.
        """
        rows = self.conn.execute("""
            SELECT card_id, skada_score, pick_rate, win_rate_delta,
                   deck_win_rate, hold_rate
            FROM cards WHERE character = ?
            ORDER BY skada_score DESC
        """, (character,)).fetchall()

        if not rows:
            return {}

        scores = [r["skada_score"] for r in rows]
        min_s, max_s = min(scores), max(scores)
        score_range = max_s - min_s if max_s > min_s else 1.0

        result = {}
        for r in rows:
            normalized = (r["skada_score"] - min_s) / score_range
            result[r["card_id"]] = {
                "score": round(normalized, 4),
                "skada_score": r["skada_score"],
                "pick_rate": r["pick_rate"],
                "win_rate_delta": r["win_rate_delta"],
                "deck_win_rate": r["deck_win_rate"],
                "hold_rate": r["hold_rate"],
            }
        return result

    def export_boss_difficulty(self):
        """Export boss difficulty rankings for AI reward shaping."""
        bosses = self.boss_guide()
        result = {}
        for b in bosses:
            result[b["encounter"]] = {
                "wipe_rate": b["wipe_rate"],
                "win_avg_dpt": b["win_avg_dpt"],
                "lose_avg_dpt": b["lose_avg_dpt"],
                "best_cards": [c["card_id"] for c in b.get("best_cards", [])[:5]],
            }
        return result

    def export_encounter_difficulty(self):
        """Export encounter difficulty for AI path planning."""
        rows = self.conn.execute("""
            SELECT encounter, enc_type, avg_damage_taken, wipe_rate
            FROM encounters
            ORDER BY wipe_rate DESC
        """).fetchall()
        return {r["encounter"]: {
            "enc_type": r["enc_type"],
            "avg_damage_taken": r["avg_damage_taken"],
            "wipe_rate": r["wipe_rate"],
        } for r in rows}


def main():
    parser = argparse.ArgumentParser(description="Query Skada Analytics data")
    parser.add_argument("command", choices=[
        "card-tier", "card-info", "synergies", "best-picks",
        "relic-tier", "encounter-danger", "boss-guide",
        "deck-size", "overview", "run-stats", "death-causes",
        "export-card-priors", "export-boss-difficulty",
    ])
    parser.add_argument("args", nargs="*", help="Command arguments")
    parser.add_argument("--floor", default="early", choices=["early", "mid", "late"])
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    sq = SkadaQuery(args.db)
    character = args.args[0] if args.args else None

    commands = {
        "card-tier": lambda: sq.card_tier_list(character, top_n=args.top),
        "card-info": lambda: sq.card_info(character or ""),
        "synergies": lambda: sq.card_synergies(character, top_n=args.top),
        "best-picks": lambda: sq.best_picks_by_floor(character, args.floor, args.top),
        "relic-tier": lambda: sq.relic_tier_list(top_n=args.top),
        "encounter-danger": lambda: sq.encounter_danger_list(top_n=args.top),
        "boss-guide": lambda: sq.boss_guide(character),
        "deck-size": lambda: sq.optimal_deck_size(character or "IRONCLAD"),
        "overview": lambda: sq.overview(),
        "run-stats": lambda: sq.run_stats(character),
        "death-causes": lambda: sq.death_cause_stats(character, top_n=args.top),
        "export-card-priors": lambda: sq.export_card_priors(character or "IRONCLAD"),
        "export-boss-difficulty": lambda: sq.export_boss_difficulty(),
    }

    result = commands[args.command]()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    sq.close()


if __name__ == "__main__":
    main()
