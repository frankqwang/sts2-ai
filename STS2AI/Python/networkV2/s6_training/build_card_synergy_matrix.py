#!/usr/bin/env python3
"""从 skada 历史数据挖 card-pair synergy lift,产出 skada_card_synergy.sqlite。

数据源:
  Primary: skada_analytics.sqlite(有 is_victory,能算 precise conditional win rate)
    - run_final_deck(~17K rows,~850 unique runs 含 deck)
    - runs(19417,is_victory 分布 42/58)
  Secondary(TODO):runs_victory jsonl 的 final_deck(~18K victory runs)作为
    co-occurrence 频次 augment。

Synergy lift 定义:
  对每对 (A, B) 在**同一 deck** 出现:
    freq_runs        = n runs having both
    winrate_both     = wins / total among those runs
    winrate_baseline = 全局该 character 的 base win rate
    lift             = winrate_both - winrate_baseline
    smoothed_lift    = lift × sigmoid((freq_runs - 5) / 5)  # 样本少时抑到 0

  另外算一个 pointwise mutual information(PMI) style 指标:
    cooccur_rate     = P(A and B in deck) / (P(A) × P(B))  # >1 = 比独立共现更频繁
  既不用依赖 win/loss,也给**频率 synergy**(共现频率高 → 经验搭配)的信号。

输出 schema:
  card_pair_synergy(
      character TEXT,           -- 分角色(不同池子 synergy 差异大)
      card_a    TEXT,
      card_b    TEXT,
      n_runs            INTEGER,
      winrate_both      REAL,
      winrate_baseline  REAL,
      lift              REAL,        -- precise(win/loss 数据)
      smoothed_lift     REAL,        -- 低样本时抑到 0
      cooccur_pmi       REAL,        -- 频次 synergy(不依赖 win/loss)
      PRIMARY KEY (character, card_a, card_b)
  )
  约束:card_a < card_b(字典序,保证无重复对)

用法:
  python -m networkV2.s6_training.build_card_synergy_matrix \
      --db STS2AI/Assets/datasets/skada/skada_analytics.sqlite \
      --output ../Artifacts/skada_card_synergy.sqlite \
      --min-pair-runs 3
"""
from __future__ import annotations

import argparse
import itertools
import logging
import math
import sqlite3
from collections import defaultdict
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _normalize_card_id(cid: str) -> str:
    """skada 里 card_id 有 UPPER 和 lower 混用,统一 lower + 去 + 升级后缀。"""
    s = str(cid or "").strip().lower()
    while s.endswith("+"):
        s = s[:-1]
    return s


def build(
    db_path: Path,
    output_db: Path,
    *,
    min_pair_runs: int = 3,
    min_card_in_runs: int = 5,
) -> dict:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    logger.info(f"loading deck data from {db_path}...")

    # 每个 run 的 deck set(去重后的 card_id 集合,不管 count)
    run_decks: dict[int, set[str]] = defaultdict(set)
    run_meta: dict[int, tuple[str, int]] = {}  # run_id → (character, is_victory)

    # 先载 runs meta
    for r in con.execute("SELECT run_id, character, is_victory FROM runs"):
        run_meta[int(r["run_id"])] = (
            str(r["character"] or "").upper(),
            int(r["is_victory"] or 0),
        )

    # 载每个 run 的 deck
    n_deck_rows = 0
    for r in con.execute("SELECT run_id, card_id FROM run_final_deck"):
        rid = int(r["run_id"])
        if rid not in run_meta:
            continue
        cid = _normalize_card_id(r["card_id"])
        if cid:
            run_decks[rid].add(cid)
            n_deck_rows += 1
    logger.info(
        f"loaded {n_deck_rows} deck rows covering {len(run_decks)} runs "
        f"(skipped {sum(1 for rid in run_decks if rid not in run_meta)} orphans)"
    )

    # 按 character 分桶
    # (character, card_a, card_b) → [total_runs, wins]
    pair_stats: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    card_count: dict[tuple[str, str], int] = defaultdict(int)   # (character, card_id) → n runs containing it
    char_run_count: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # (character,) → [total, wins]

    for rid, deck in run_decks.items():
        ch, victory = run_meta.get(rid, ("", 0))
        if not ch:
            continue
        char_run_count[ch][0] += 1
        char_run_count[ch][1] += victory

        cards = sorted(deck)
        # single counts
        for c in cards:
            card_count[(ch, c)] += 1
        # pairs
        for a, b in itertools.combinations(cards, 2):
            # a < b 已由 sorted + combinations 保证
            s = pair_stats[(ch, a, b)]
            s[0] += 1
            s[1] += victory

    # 构造 sqlite output
    output_db.parent.mkdir(parents=True, exist_ok=True)
    if output_db.exists():
        output_db.unlink()
    out = sqlite3.connect(str(output_db))
    out.execute("""
        CREATE TABLE card_pair_synergy (
            character         TEXT NOT NULL,
            card_a            TEXT NOT NULL,
            card_b            TEXT NOT NULL,
            n_runs            INTEGER NOT NULL,
            winrate_both      REAL NOT NULL,
            winrate_baseline  REAL NOT NULL,
            lift              REAL NOT NULL,
            smoothed_lift     REAL NOT NULL,
            cooccur_pmi       REAL NOT NULL,
            PRIMARY KEY (character, card_a, card_b)
        )
    """)
    out.execute("CREATE INDEX idx_char_a ON card_pair_synergy(character, card_a)")
    out.execute("CREATE TABLE metadata (key TEXT, value TEXT)")

    rows = []
    n_kept = 0
    for (ch, a, b), (total, wins) in pair_stats.items():
        if total < min_pair_runs:
            continue
        # baseline = 该 character 的总 victory rate
        char_total, char_wins = char_run_count[ch]
        baseline = char_wins / max(char_total, 1)
        winrate_both = wins / max(total, 1)
        lift = winrate_both - baseline
        # smoothed:样本少(<5 次)抑到 0
        conf = 1.0 / (1.0 + math.exp(-(total - 5) / 5.0))
        smoothed_lift = lift * conf

        # PMI:card pair 共现频率 vs 独立预期
        n_ch = char_total
        p_a = card_count[(ch, a)] / max(n_ch, 1)
        p_b = card_count[(ch, b)] / max(n_ch, 1)
        p_ab = total / max(n_ch, 1)
        # NPMI = log(p_ab / (p_a p_b)) / -log(p_ab),clamp 到 [-1, 1]
        if p_ab > 0 and p_a > 0 and p_b > 0:
            pmi = math.log(p_ab / (p_a * p_b))
            denom = -math.log(p_ab) if p_ab < 1.0 else 1e-6
            npmi = max(-1.0, min(1.0, pmi / denom))
        else:
            npmi = 0.0

        rows.append((ch, a, b, total, winrate_both, baseline, lift, smoothed_lift, npmi))
        n_kept += 1

    logger.info(
        f"mined {len(pair_stats)} unique (char, pair),"
        f" kept {n_kept} with n_runs>={min_pair_runs}"
    )
    out.executemany(
        "INSERT INTO card_pair_synergy VALUES (?,?,?,?,?,?,?,?,?)",
        rows,
    )

    # metadata
    from datetime import datetime, timezone
    out.executemany(
        "INSERT INTO metadata VALUES (?, ?)",
        [
            ("schema_version", "1"),
            ("source", str(db_path)),
            ("generated_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
            ("min_pair_runs", str(min_pair_runs)),
            ("n_kept_pairs", str(n_kept)),
            ("n_characters", str(len(char_run_count))),
        ],
    )
    out.commit()
    out.close()
    con.close()

    return {
        "runs_with_deck": len(run_decks),
        "pairs_total": len(pair_stats),
        "pairs_kept": n_kept,
        "characters": dict(
            (ch, f"{w}/{t} = {w/max(t,1)*100:.1f}%") for ch, (t, w) in char_run_count.items()
        ),
        "output_db": str(output_db),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path,
                   default=Path("STS2AI/Assets/datasets/skada/skada_analytics.sqlite"),
                   help="skada_analytics.sqlite 路径(含 is_victory 的旧库)")
    p.add_argument("--output", type=Path,
                   default=Path("../Artifacts/skada_card_synergy.sqlite"))
    p.add_argument("--min-pair-runs", type=int, default=3)
    args = p.parse_args()

    stats = build(args.db, args.output, min_pair_runs=args.min_pair_runs)
    print()
    print("=== synergy matrix summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
