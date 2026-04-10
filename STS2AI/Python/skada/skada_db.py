"""SQLite schema and upsert helpers for the checked-in Skada dataset."""

from pathlib import Path
import sqlite3

DB_PATH = Path(__file__).resolve().parents[2] / "Assets" / "datasets" / "skada" / "skada_analytics.sqlite"


def get_connection(db_path=None):
    resolved_path = Path(db_path) if db_path is not None else DB_PATH
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(resolved_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def create_tables(conn: sqlite3.Connection):
    c = conn.cursor()

    # ── Cards ──
    c.execute("""
    CREATE TABLE IF NOT EXISTS cards (
        card_id         TEXT PRIMARY KEY,
        character       TEXT,
        card_pool       TEXT,
        name_en         TEXT,
        name_zh         TEXT,
        rank            INTEGER,
        skada_score     REAL,
        pick_rate       REAL,
        win_rate_picked REAL,
        win_rate_skipped REAL,
        win_rate_delta  REAL,
        delta_vs_skipped REAL,
        deck_win_rate   REAL,
        deck_runs       INTEGER,
        hold_rate       REAL,
        seen            INTEGER,
        dmg_per_play    REAL,
        blk_per_play    REAL,
        dmg_per_energy  REAL,
        plays_per_combat REAL,
        obtain_count    INTEGER,
        obtain_runs     INTEGER,
        obtain_win_rate REAL,
        avg_obtain_floor REAL
    )""")

    # ── Card obtain sources ──
    c.execute("""
    CREATE TABLE IF NOT EXISTS card_obtain_sources (
        card_id         TEXT,
        source_type     TEXT,
        acquisitions    INTEGER,
        runs            INTEGER,
        win_rate        REAL,
        PRIMARY KEY (card_id, source_type),
        FOREIGN KEY (card_id) REFERENCES cards(card_id)
    )""")

    # ── Card floor value (early/mid/late pick rates & win rates) ──
    c.execute("""
    CREATE TABLE IF NOT EXISTS card_floor_value (
        card_id         TEXT,
        character       TEXT,
        stage           TEXT,   -- 'early', 'mid', 'late'
        pick_rate       REAL,
        win_rate_picked REAL,
        win_rate_skipped REAL,
        delta_vs_community REAL,
        seen            INTEGER,
        picked          INTEGER,
        confidence      TEXT,
        PRIMARY KEY (card_id, stage)
    )""")

    # ── Card companions (synergies) ──
    c.execute("""
    CREATE TABLE IF NOT EXISTS card_companions (
        card_a          TEXT,
        card_b          TEXT,
        character       TEXT,
        pair_runs       INTEGER,
        pair_win_rate   REAL,
        card_a_solo_wr  REAL,
        card_b_solo_wr  REAL,
        synergy_lift    REAL,
        confidence      TEXT,
        PRIMARY KEY (card_a, card_b, character)
    )""")

    # ── Card upgrade value ──
    c.execute("""
    CREATE TABLE IF NOT EXISTS card_upgrade_value (
        card_id         TEXT PRIMARY KEY,
        name_en         TEXT,
        name_zh         TEXT,
        total_upgrades  INTEGER,
        avg_position    REAL
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS card_upgrade_by_position (
        card_id         TEXT,
        position        INTEGER,
        count           INTEGER,
        win_rate        REAL,
        PRIMARY KEY (card_id, position),
        FOREIGN KEY (card_id) REFERENCES card_upgrade_value(card_id)
    )""")

    # ── Relics ──
    c.execute("""
    CREATE TABLE IF NOT EXISTS relics (
        relic_id        TEXT PRIMARY KEY,
        name_en         TEXT,
        name_zh         TEXT,
        seen            INTEGER,
        times_offered   INTEGER,
        times_picked    INTEGER,
        times_skipped   INTEGER,
        pick_rate       REAL,
        skip_rate       REAL,
        win_rate_picked REAL,
        win_rate_skipped REAL,
        times_owned     INTEGER,
        hold_rate       REAL,
        win_rate_owned  REAL
    )""")

    # ── Encounters ──
    c.execute("""
    CREATE TABLE IF NOT EXISTS encounters (
        encounter       TEXT PRIMARY KEY,
        enc_type        TEXT,
        type            TEXT,
        name_en         TEXT,
        name_zh         TEXT,
        times_seen      INTEGER,
        avg_turns       REAL,
        avg_damage_taken REAL,
        avg_dpt         REAL,
        wipe_rate       REAL
    )""")

    # ── Boss guide ──
    c.execute("""
    CREATE TABLE IF NOT EXISTS boss_guide (
        encounter       TEXT PRIMARY KEY,
        name_en         TEXT,
        name_zh         TEXT,
        win_avg_dpt     REAL,
        lose_avg_dpt    REAL,
        wipe_rate       REAL
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS boss_best_cards (
        encounter       TEXT,
        card_id         TEXT,
        name_en         TEXT,
        name_zh         TEXT,
        dmg_per_play    REAL,
        plays           INTEGER,
        PRIMARY KEY (encounter, card_id),
        FOREIGN KEY (encounter) REFERENCES boss_guide(encounter)
    )""")

    # ── Runs ──
    c.execute("""
    CREATE TABLE IF NOT EXISTS runs (
        run_id          INTEGER PRIMARY KEY,
        character       TEXT,
        ascension       INTEGER,
        seed            TEXT,
        is_victory      INTEGER,
        abandoned       INTEGER,
        death_cause     TEXT,
        floor_reached   INTEGER,
        duration_sec    INTEGER,
        player_count    INTEGER,
        game_version    TEXT,
        created_at      TEXT,
        player_name     TEXT
    )""")

    # ── Decisions (campfire, etc.) ──
    c.execute("""
    CREATE TABLE IF NOT EXISTS campfire_decisions (
        action          TEXT PRIMARY KEY,
        name_en         TEXT,
        name_zh         TEXT,
        count           INTEGER,
        usage_rate      REAL,
        win_rate        REAL
    )""")

    # ── Deck size curves ──
    c.execute("""
    CREATE TABLE IF NOT EXISTS deck_size_curves (
        character       TEXT,
        deck_size       INTEGER,
        total           INTEGER,
        wins            INTEGER,
        win_rate        REAL,
        PRIMARY KEY (character, deck_size)
    )""")

    # ── Overview stats ──
    c.execute("""
    CREATE TABLE IF NOT EXISTS overview (
        key             TEXT PRIMARY KEY,
        value           REAL
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS character_win_rates (
        character       TEXT PRIMARY KEY,
        name_en         TEXT,
        name_zh         TEXT,
        total_runs      INTEGER,
        wins            INTEGER,
        win_rate        REAL
    )""")

    # ── Metadata ──
    c.execute("""
    CREATE TABLE IF NOT EXISTS scrape_meta (
        endpoint        TEXT PRIMARY KEY,
        last_scraped    TEXT,
        record_count    INTEGER
    )""")

    conn.commit()


# ── Bulk insert helpers ──

def upsert_cards(conn, cards):
    c = conn.cursor()
    for card in cards:
        dn = card.get("display_name", {})
        c.execute("""
        INSERT OR REPLACE INTO cards VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            card["card_id"], card.get("character"), card.get("card_pool"),
            dn.get("en"), dn.get("zh"),
            card.get("rank"), card.get("skada_score"),
            card.get("pick_rate"), card.get("win_rate_picked"), card.get("win_rate_skipped"),
            card.get("win_rate_delta"), card.get("delta_vs_skipped"),
            card.get("deck_win_rate"), card.get("deck_runs"),
            card.get("hold_rate"), card.get("seen"),
            card.get("dmg_per_play"), card.get("blk_per_play"),
            card.get("dmg_per_energy"), card.get("plays_per_combat"),
            card.get("obtain_count"), card.get("obtain_runs"),
            card.get("obtain_win_rate"), card.get("avg_obtain_floor"),
        ))
        for src in card.get("obtain_sources", []):
            c.execute("""
            INSERT OR REPLACE INTO card_obtain_sources VALUES (?,?,?,?,?)
            """, (card["card_id"], src["source_type"], src["acquisitions"],
                  src["runs"], src["win_rate"]))
    conn.commit()


def upsert_card_floor_value(conn, items):
    c = conn.cursor()
    for item in items:
        card_id = item["card_id"]
        character = item.get("character", "")
        for stage_name, stage_data in item.get("stages", {}).items():
            if stage_data is None:
                continue
            c.execute("""
            INSERT OR REPLACE INTO card_floor_value VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                card_id, character, stage_name,
                stage_data.get("pick_rate"),
                stage_data.get("win_rate_picked"),
                stage_data.get("win_rate_skipped"),
                stage_data.get("delta_vs_community"),
                stage_data.get("seen"),
                stage_data.get("picked"),
                stage_data.get("confidence"),
            ))
    conn.commit()


def upsert_card_companions(conn, pairs):
    c = conn.cursor()
    for pair in pairs:
        c.execute("""
        INSERT OR REPLACE INTO card_companions VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            pair["card_a"], pair["card_b"], pair.get("character"),
            pair.get("pair_runs"), pair.get("pair_win_rate"),
            pair.get("card_a_solo_wr"), pair.get("card_b_solo_wr"),
            pair.get("synergy_lift"), pair.get("confidence"),
        ))
    conn.commit()


def upsert_card_upgrade_value(conn, rows):
    c = conn.cursor()
    for row in rows:
        dn = row.get("display_name", row.get("card_name", {}))
        c.execute("""
        INSERT OR REPLACE INTO card_upgrade_value VALUES (?,?,?,?,?)
        """, (
            row["card_id"],
            dn.get("en"), dn.get("zh"),
            row.get("total_upgrades"), row.get("avg_position"),
        ))
        for pos in row.get("by_position", []):
            c.execute("""
            INSERT OR REPLACE INTO card_upgrade_by_position VALUES (?,?,?,?)
            """, (row["card_id"], pos["position"], pos["count"], pos["win_rate"]))
    conn.commit()


def upsert_relics(conn, relics):
    c = conn.cursor()
    for r in relics:
        dn = r.get("display_name", {})
        c.execute("""
        INSERT OR REPLACE INTO relics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            r["relic_id"],
            dn.get("en"), dn.get("zh"),
            r.get("seen"), r.get("times_offered"), r.get("times_picked"),
            r.get("times_skipped"), r.get("pick_rate"), r.get("skip_rate"),
            r.get("win_rate_picked"), r.get("win_rate_skipped"),
            r.get("times_owned"), r.get("hold_rate"), r.get("win_rate_owned"),
        ))
    conn.commit()


def upsert_encounters(conn, encounters):
    c = conn.cursor()
    for e in encounters:
        dn = e.get("display_name", {})
        c.execute("""
        INSERT OR REPLACE INTO encounters VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            e["encounter"], e.get("enc_type"), e.get("type"),
            dn.get("en"), dn.get("zh"),
            e.get("times_seen"), e.get("avg_turns"),
            e.get("avg_damage_taken"), e.get("avg_dpt"), e.get("wipe_rate"),
        ))
    conn.commit()


def upsert_boss_guide(conn, bosses):
    c = conn.cursor()
    for b in bosses:
        dn = b.get("display_name", {})
        c.execute("""
        INSERT OR REPLACE INTO boss_guide VALUES (?,?,?,?,?,?)
        """, (
            b["encounter"],
            dn.get("en"), dn.get("zh"),
            b.get("win_avg_dpt"), b.get("lose_avg_dpt"), b.get("wipe_rate"),
        ))
        for card in b.get("best_cards", []):
            cdn = card.get("display_name", {})
            c.execute("""
            INSERT OR REPLACE INTO boss_best_cards VALUES (?,?,?,?,?,?)
            """, (
                b["encounter"], card["card_id"],
                cdn.get("en"), cdn.get("zh"),
                card.get("dmg_per_play"), card.get("plays"),
            ))
    conn.commit()


def upsert_runs(conn, runs):
    c = conn.cursor()
    for r in runs:
        c.execute("""
        INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            r["run_id"], r.get("character"), r.get("ascension"),
            r.get("seed"), 1 if r.get("is_victory") else 0,
            1 if r.get("abandoned") else 0,
            r.get("death_cause"), r.get("floor_reached"),
            r.get("duration_sec"), r.get("player_count"),
            r.get("game_version"), r.get("created_at"),
            r.get("player_name"),
        ))
    conn.commit()


def upsert_campfire_decisions(conn, decisions):
    c = conn.cursor()
    for d in decisions:
        dn = d.get("display_name", {})
        c.execute("""
        INSERT OR REPLACE INTO campfire_decisions VALUES (?,?,?,?,?,?)
        """, (
            d["action"],
            dn.get("en"), dn.get("zh"),
            d.get("count"), d.get("usage_rate"), d.get("win_rate"),
        ))
    conn.commit()


def upsert_deck_size_curves(conn, characters):
    c = conn.cursor()
    for ch in characters:
        character = ch["character"]
        for pt in ch.get("curve", []):
            c.execute("""
            INSERT OR REPLACE INTO deck_size_curves VALUES (?,?,?,?,?)
            """, (character, pt["deck_size"], pt["total"], pt["wins"], pt["win_rate"]))
    conn.commit()


def upsert_overview(conn, data):
    c = conn.cursor()
    for key in ("total_runs", "total_combats", "win_rate", "average_floor",
                "avg_time", "unique_players", "total_play_hours"):
        if key in data:
            c.execute("INSERT OR REPLACE INTO overview VALUES (?,?)", (key, data[key]))
    for ch in data.get("character_win_rates", []):
        dn = ch.get("display_name", {})
        c.execute("""
        INSERT OR REPLACE INTO character_win_rates VALUES (?,?,?,?,?,?)
        """, (
            ch["character"], dn.get("en"), dn.get("zh"),
            ch.get("total_runs"), ch.get("wins"), ch.get("win_rate"),
        ))
    conn.commit()


def update_scrape_meta(conn, endpoint, count):
    from datetime import datetime
    conn.execute("""
    INSERT OR REPLACE INTO scrape_meta VALUES (?,?,?)
    """, (endpoint, datetime.now().isoformat(), count))
    conn.commit()


if __name__ == "__main__":
    conn = get_connection()
    create_tables(conn)
    print(f"Database created at {DB_PATH}")
    conn.close()
