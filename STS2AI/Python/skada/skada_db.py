"""SQLite schema and upsert helpers for the checked-in Skada dataset."""

from pathlib import Path
import json
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

    c.execute("""
    CREATE TABLE IF NOT EXISTS run_details (
        run_id              INTEGER PRIMARY KEY,
        status              TEXT NOT NULL,
        scraped_at          TEXT,
        last_error          TEXT,
        total_combats       INTEGER,
        total_floors        INTEGER,
        final_deck_count    INTEGER,
        final_relic_count   INTEGER,
        map_act_count       INTEGER,
        total_damage_taken  INTEGER,
        total_damage_dealt  INTEGER,
        net_gold_change     INTEGER,
        picked_card_count   INTEGER,
        offered_card_count  INTEGER,
        raw_json            TEXT,
        FOREIGN KEY (run_id) REFERENCES runs(run_id)
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS run_combats (
        combat_id            INTEGER PRIMARY KEY,
        run_id               INTEGER,
        floor                INTEGER,
        encounter            TEXT,
        enc_type             TEXT,
        type                 TEXT,
        turns                INTEGER,
        won                  INTEGER,
        name_en              TEXT,
        name_zh              TEXT,
        total_dmg_dealt      INTEGER,
        total_dmg_taken      INTEGER,
        raw_json             TEXT,
        FOREIGN KEY (run_id) REFERENCES runs(run_id)
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS run_combat_stats (
        combat_id            INTEGER,
        stat_index           INTEGER,
        character_id         TEXT,
        dmg_dealt            INTEGER,
        dmg_taken            INTEGER,
        block_gained         INTEGER,
        cards_played         INTEGER,
        energy_spent         INTEGER,
        energy_wasted        INTEGER,
        overkill             INTEGER,
        max_hit              INTEGER,
        assist_dmg           INTEGER,
        dmg_prevented        INTEGER,
        potions_used         INTEGER,
        name_en              TEXT,
        name_zh              TEXT,
        raw_json             TEXT,
        PRIMARY KEY (combat_id, stat_index),
        FOREIGN KEY (combat_id) REFERENCES run_combats(combat_id)
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS run_combat_card_perf (
        combat_id            INTEGER,
        perf_index           INTEGER,
        card_id              TEXT,
        character_id         TEXT,
        damage               INTEGER,
        block                INTEGER,
        energy               INTEGER,
        plays                INTEGER,
        name_en              TEXT,
        name_zh              TEXT,
        raw_json             TEXT,
        PRIMARY KEY (combat_id, perf_index),
        FOREIGN KEY (combat_id) REFERENCES run_combats(combat_id)
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS run_floor_timeline (
        run_id               INTEGER,
        floor                INTEGER,
        room_type            TEXT,
        room_name_en         TEXT,
        room_name_zh         TEXT,
        hp_before            INTEGER,
        hp_after             INTEGER,
        gold_before          INTEGER,
        gold_after           INTEGER,
        event_text           TEXT,
        campfire_choice      TEXT,
        campfire_name_en     TEXT,
        campfire_name_zh     TEXT,
        combat_id            INTEGER,
        encounter            TEXT,
        enc_type             TEXT,
        turns                INTEGER,
        won                  INTEGER,
        total_dmg_dealt      INTEGER,
        total_dmg_taken      INTEGER,
        raw_json             TEXT,
        PRIMARY KEY (run_id, floor),
        FOREIGN KEY (run_id) REFERENCES runs(run_id)
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS run_floor_card_choices (
        run_id               INTEGER,
        floor                INTEGER,
        choice_index         INTEGER,
        card_id              TEXT,
        was_picked           INTEGER,
        name_en              TEXT,
        name_zh              TEXT,
        PRIMARY KEY (run_id, floor, choice_index),
        FOREIGN KEY (run_id) REFERENCES runs(run_id)
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS run_floor_relic_choices (
        run_id               INTEGER,
        floor                INTEGER,
        source_kind          TEXT,
        choice_index         INTEGER,
        relic_id             TEXT,
        was_picked           INTEGER,
        name_en              TEXT,
        name_zh              TEXT,
        PRIMARY KEY (run_id, floor, source_kind, choice_index),
        FOREIGN KEY (run_id) REFERENCES runs(run_id)
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS run_floor_shop_actions (
        run_id               INTEGER,
        floor                INTEGER,
        action_index         INTEGER,
        action_type          TEXT,
        item_id              TEXT,
        item_name_en         TEXT,
        item_name_zh         TEXT,
        action_name_en       TEXT,
        action_name_zh       TEXT,
        raw_json             TEXT,
        PRIMARY KEY (run_id, floor, action_index),
        FOREIGN KEY (run_id) REFERENCES runs(run_id)
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS run_card_upgrades (
        run_id               INTEGER,
        floor                INTEGER,
        upgrade_index        INTEGER,
        card_id              TEXT,
        name_en              TEXT,
        name_zh              TEXT,
        PRIMARY KEY (run_id, floor, upgrade_index),
        FOREIGN KEY (run_id) REFERENCES runs(run_id)
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS run_final_deck (
        run_id               INTEGER,
        deck_index           INTEGER,
        card_id              TEXT,
        count                INTEGER,
        name_en              TEXT,
        name_zh              TEXT,
        PRIMARY KEY (run_id, deck_index),
        FOREIGN KEY (run_id) REFERENCES runs(run_id)
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS run_final_relics (
        run_id               INTEGER,
        relic_index          INTEGER,
        relic_id             TEXT,
        name_en              TEXT,
        name_zh              TEXT,
        PRIMARY KEY (run_id, relic_index),
        FOREIGN KEY (run_id) REFERENCES runs(run_id)
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS run_map_acts (
        run_id               INTEGER,
        act                  INTEGER,
        width                INTEGER,
        height               INTEGER,
        boss_x               INTEGER,
        boss_y               INTEGER,
        start_coords_json    TEXT,
        visited_coords_json  TEXT,
        nodes_json           TEXT,
        raw_json             TEXT,
        PRIMARY KEY (run_id, act),
        FOREIGN KEY (run_id) REFERENCES runs(run_id)
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

def _display_name_parts(payload):
    if not isinstance(payload, dict):
        return None, None
    return payload.get("en"), payload.get("zh")


def _json_dumps(value):
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def record_run_detail_error(conn, run_id, error_message):
    from datetime import datetime
    conn.execute("""
    INSERT OR REPLACE INTO run_details (
        run_id, status, scraped_at, last_error, total_combats, total_floors,
        final_deck_count, final_relic_count, map_act_count, total_damage_taken,
        total_damage_dealt, net_gold_change, picked_card_count, offered_card_count, raw_json
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        run_id, "error", datetime.now().isoformat(), str(error_message)[:2000],
        None, None, None, None, None, None, None, None, None, None, None,
    ))
    conn.commit()


def get_pending_run_detail_ids(conn, limit=None, retry_errors=False):
    where = "d.run_id IS NULL"
    if retry_errors:
        where = "(d.run_id IS NULL OR d.status = 'error')"
    sql = f"""
        SELECT r.run_id
        FROM runs r
        LEFT JOIN run_details d ON d.run_id = r.run_id
        WHERE {where}
        ORDER BY r.run_id
    """
    params = []
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    return [int(row[0]) for row in conn.execute(sql, params).fetchall()]


def get_run_detail_progress(conn):
    total_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    ok_count = conn.execute("SELECT COUNT(*) FROM run_details WHERE status = 'ok'").fetchone()[0]
    error_count = conn.execute("SELECT COUNT(*) FROM run_details WHERE status = 'error'").fetchone()[0]
    pending_count = max(0, total_runs - ok_count - error_count)
    return {
        "total_runs": int(total_runs),
        "ok": int(ok_count),
        "error": int(error_count),
        "pending": int(pending_count),
    }

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


def upsert_run_detail(conn, payload):
    from datetime import datetime

    run = payload.get("run") or {}
    run_id = run.get("run_id")
    if run_id is None:
        raise ValueError("Run detail payload missing run.run_id")

    upsert_runs(conn, [run])

    combats = payload.get("combats") or []
    floor_timeline = payload.get("floor_timeline") or []
    final_deck = payload.get("final_deck") or []
    final_relics = payload.get("final_relics") or []
    map_acts = payload.get("map_acts") or []

    total_damage_taken = 0
    total_damage_dealt = 0
    for combat in combats:
        for stat in combat.get("combat_stats") or []:
            total_damage_taken += int(stat.get("dmg_taken") or 0)
            total_damage_dealt += int(stat.get("dmg_dealt") or 0)

    net_gold_change = 0
    picked_card_count = 0
    offered_card_count = 0
    for row in floor_timeline:
        before = row.get("gold_before")
        after = row.get("gold_after")
        if before is not None and after is not None:
            net_gold_change += int(after) - int(before)
        card_choices = row.get("card_choices") or []
        offered_card_count += len(card_choices)
        picked_card_count += sum(1 for choice in card_choices if choice.get("was_picked"))

    c = conn.cursor()
    c.execute("""
        DELETE FROM run_combat_stats
        WHERE combat_id IN (SELECT combat_id FROM run_combats WHERE run_id = ?)
    """, (run_id,))
    c.execute("""
        DELETE FROM run_combat_card_perf
        WHERE combat_id IN (SELECT combat_id FROM run_combats WHERE run_id = ?)
    """, (run_id,))
    for table in (
        "run_combats",
        "run_floor_timeline",
        "run_floor_card_choices",
        "run_floor_relic_choices",
        "run_floor_shop_actions",
        "run_card_upgrades",
        "run_final_deck",
        "run_final_relics",
        "run_map_acts",
        "run_details",
    ):
        c.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))

    c.execute("""
        INSERT OR REPLACE INTO run_details VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        run_id,
        "ok",
        datetime.now().isoformat(),
        None,
        len(combats),
        len(floor_timeline),
        len(final_deck),
        len(final_relics),
        len(map_acts),
        total_damage_taken,
        total_damage_dealt,
        net_gold_change,
        picked_card_count,
        offered_card_count,
        _json_dumps(payload),
    ))

    for combat in combats:
        name_en, name_zh = _display_name_parts(
            combat.get("encounter_display_name")
            or combat.get("display_name")
            or combat.get("encounter_name")
        )
        combat_summary = next(
            (row for row in floor_timeline if (row.get("combat") or {}).get("combat_id") == combat.get("combat_id")),
            None,
        )
        combat_blob = combat_summary.get("combat") if isinstance(combat_summary, dict) else None
        c.execute("""
            INSERT OR REPLACE INTO run_combats VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            combat.get("combat_id"),
            run_id,
            combat.get("floor"),
            combat.get("encounter"),
            combat.get("enc_type"),
            combat.get("type"),
            combat.get("turns"),
            1 if combat.get("won") else 0,
            name_en,
            name_zh,
            (combat_blob or {}).get("total_dmg_dealt"),
            (combat_blob or {}).get("total_dmg_taken"),
            _json_dumps(combat),
        ))

        for stat_index, stat in enumerate(combat.get("combat_stats") or []):
            stat_en, stat_zh = _display_name_parts(
                stat.get("character_display_name") or stat.get("display_name") or stat.get("character_name")
            )
            c.execute("""
                INSERT OR REPLACE INTO run_combat_stats VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                combat.get("combat_id"),
                stat_index,
                stat.get("character_id"),
                stat.get("dmg_dealt"),
                stat.get("dmg_taken"),
                stat.get("block_gained"),
                stat.get("cards_played"),
                stat.get("energy_spent"),
                stat.get("energy_wasted"),
                stat.get("overkill"),
                stat.get("max_hit"),
                stat.get("assist_dmg"),
                stat.get("dmg_prevented"),
                stat.get("potions_used"),
                stat_en,
                stat_zh,
                _json_dumps(stat),
            ))

        for perf_index, perf in enumerate(combat.get("card_combat_perf") or []):
            perf_en, perf_zh = _display_name_parts(
                perf.get("display_name") or perf.get("card_name")
            )
            c.execute("""
                INSERT OR REPLACE INTO run_combat_card_perf VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                combat.get("combat_id"),
                perf_index,
                perf.get("card_id"),
                perf.get("character_id"),
                perf.get("damage"),
                perf.get("block"),
                perf.get("energy"),
                perf.get("plays"),
                perf_en,
                perf_zh,
                _json_dumps(perf),
            ))

    for row in floor_timeline:
        floor = row.get("floor")
        room_en, room_zh = _display_name_parts(row.get("room_type_display_name"))
        campfire_en, campfire_zh = _display_name_parts(row.get("campfire_display_name"))
        combat = row.get("combat") or {}
        c.execute("""
            INSERT OR REPLACE INTO run_floor_timeline VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            run_id,
            floor,
            row.get("room_type"),
            room_en,
            room_zh,
            row.get("hp_before"),
            row.get("hp_after"),
            row.get("gold_before"),
            row.get("gold_after"),
            row.get("event_text"),
            row.get("campfire_choice"),
            campfire_en,
            campfire_zh,
            combat.get("combat_id"),
            combat.get("encounter"),
            combat.get("enc_type"),
            combat.get("turns"),
            1 if combat.get("won") else 0 if "won" in combat else None,
            combat.get("total_dmg_dealt"),
            combat.get("total_dmg_taken"),
            _json_dumps(row),
        ))

        for choice_index, choice in enumerate(row.get("card_choices") or []):
            choice_en, choice_zh = _display_name_parts(choice.get("display_name"))
            c.execute("""
                INSERT OR REPLACE INTO run_floor_card_choices VALUES (?,?,?,?,?,?,?)
            """, (
                run_id,
                floor,
                choice_index,
                choice.get("card_id"),
                1 if choice.get("was_picked") else 0,
                choice_en,
                choice_zh,
            ))

        for source_kind in ("relic_choices", "ancient_choices"):
            for choice_index, choice in enumerate(row.get(source_kind) or []):
                choice_en, choice_zh = _display_name_parts(choice.get("display_name"))
                c.execute("""
                    INSERT OR REPLACE INTO run_floor_relic_choices VALUES (?,?,?,?,?,?,?,?)
                """, (
                    run_id,
                    floor,
                    source_kind,
                    choice_index,
                    choice.get("relic_id"),
                    1 if choice.get("was_picked") else 0,
                    choice_en,
                    choice_zh,
                ))

        for action_index, action in enumerate(row.get("shop_actions") or []):
            item_en, item_zh = _display_name_parts(action.get("display_name"))
            action_en, action_zh = _display_name_parts(action.get("action_display_name"))
            c.execute("""
                INSERT OR REPLACE INTO run_floor_shop_actions VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                run_id,
                floor,
                action_index,
                action.get("action_type"),
                action.get("item_id"),
                item_en,
                item_zh,
                action_en,
                action_zh,
                _json_dumps(action),
            ))

        for upgrade_index, upgrade in enumerate(row.get("card_upgrades") or []):
            upgrade_en, upgrade_zh = _display_name_parts(upgrade.get("display_name"))
            c.execute("""
                INSERT OR REPLACE INTO run_card_upgrades VALUES (?,?,?,?,?,?)
            """, (
                run_id,
                floor,
                upgrade_index,
                upgrade.get("card_id"),
                upgrade_en,
                upgrade_zh,
            ))

    for deck_index, card in enumerate(final_deck):
        card_en, card_zh = _display_name_parts(card.get("display_name"))
        c.execute("""
            INSERT OR REPLACE INTO run_final_deck VALUES (?,?,?,?,?,?)
        """, (
            run_id,
            deck_index,
            card.get("card_id"),
            card.get("count"),
            card_en,
            card_zh,
        ))

    for relic_index, relic in enumerate(final_relics):
        relic_en, relic_zh = _display_name_parts(relic.get("display_name"))
        c.execute("""
            INSERT OR REPLACE INTO run_final_relics VALUES (?,?,?,?,?)
        """, (
            run_id,
            relic_index,
            relic.get("relic_id"),
            relic_en,
            relic_zh,
        ))

    for act_payload in map_acts:
        boss = act_payload.get("boss") or [None, None]
        c.execute("""
            INSERT OR REPLACE INTO run_map_acts VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            run_id,
            act_payload.get("act"),
            act_payload.get("width"),
            act_payload.get("height"),
            boss[0] if len(boss) > 0 else None,
            boss[1] if len(boss) > 1 else None,
            _json_dumps(act_payload.get("start_coords")),
            _json_dumps(act_payload.get("visited_coords")),
            _json_dumps(act_payload.get("nodes")),
            _json_dumps(act_payload),
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
