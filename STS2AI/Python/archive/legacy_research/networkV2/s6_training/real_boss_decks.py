"""从 combat_teacher artifacts 提取真实 boss-ready deck 作为 co-trainer 的训练 deck。

背景：
  synthetic `buffed_ironclad_deck` 无法模拟真实 mid-late game deck（一般 14-22 张
  + 2-3 relics）。导致 agent 对 boss 练习时"没合适 deck 就必输"，陷入
  zero-positive-advantage trap。

数据源：
  `Artifacts/combat_teacher/tactical_v1_replay_*` —— 真实 AI 训练出的 solver
  跑到 act1 boss 时 dump 的完整 state，含 deck + relics。

用法：
  from networkV2.s6_training.real_boss_decks import load_real_boss_decks
  decks = load_real_boss_decks()  # list of {deck: [...], relics: [...], ...}
  # 训练时随机抽一个给 boss 训练
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# combat_teacher artifacts 位置
_ARTIFACT_ROOT = Path(__file__).resolve().parents[3] / "Artifacts" / "combat_teacher"

# 这些数据集包含真实 boss/elite 状态的 deck snapshot
_DATASETS = [
    "tactical_v1_replay_20260415_203730/ironclad_act1_tactical_teacher_v1_replay.jsonl",
    "tactical_v1_bossfocus_20260415_200832/ironclad_act1_tactical_teacher_v1_bossfocus.jsonl",
    "tactical_v1_bossfocus_v2_20260415_201426/ironclad_act1_tactical_teacher_v1_bossfocus_v2.jsonl",
]


def _extract_deck_from_sample(sample: dict) -> dict[str, Any] | None:
    """从一个 teacher sample 提取 (deck, relics, hp, gold) 作为 co-trainer reset 的 build spec。"""
    state = sample.get("state") or {}
    player = state.get("player") or {}
    deck = player.get("deck") or []
    if len(deck) < 8:   # 至少 starter size
        return None
    card_ids = [c.get("id") or c.get("card_id") for c in deck if isinstance(c, dict)]
    card_ids = [c for c in card_ids if c]
    if not card_ids:
        return None
    relics = player.get("relics") or []
    relic_ids = [r.get("id") for r in relics if isinstance(r, dict) and r.get("id")]
    return {
        "deck": card_ids,
        "relics": relic_ids,
        "max_hp": int(player.get("max_hp", 80) or 80),
        "current_hp": int(player.get("max_hp", 80) or 80),  # 满血开始
        "gold": int(player.get("gold", 99) or 99),
        "max_energy": int(player.get("max_energy", 3) or 3),
        # metadata
        "_floor": state.get("run", {}).get("floor", 0),
        "_state_type": state.get("state_type", ""),
    }


def _dedupe_decks(decks: list[dict]) -> list[dict]:
    """按 deck 组成 + relics 去重。"""
    seen: dict[tuple, dict] = {}
    for d in decks:
        key = (
            tuple(sorted(Counter(d["deck"]).items())),
            tuple(sorted(d.get("relics", []))),
        )
        if key not in seen:
            seen[key] = d
    return list(seen.values())


def load_real_boss_decks(
    state_types: tuple[str, ...] = ("boss", "elite"),
    validate_cards: bool = True,
) -> list[dict[str, Any]]:
    """加载真实 boss-ready decks。

    Args:
      state_types: 只选 state_type 在此集合里的 sample（默认 boss+elite）
      validate_cards: True 时用 GAME_CATALOG 剔除 STS2 不存在的卡

    Returns:
      List of deck specs ({deck, relics, max_hp, ...}) ready for sim.reset(build=...)
    """
    if not _ARTIFACT_ROOT.exists():
        logger.warning(f"Artifacts not found: {_ARTIFACT_ROOT}; real_boss_decks empty")
        return []

    all_decks: list[dict] = []
    for rel in _DATASETS:
        p = _ARTIFACT_ROOT / rel
        if not p.exists():
            continue
        try:
            with p.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        s = json.loads(line)
                    except Exception:
                        continue
                    st = (s.get("state") or {}).get("state_type", "")
                    if state_types and st not in state_types:
                        continue
                    d = _extract_deck_from_sample(s)
                    if d:
                        all_decks.append(d)
        except Exception as e:
            logger.warning(f"Failed reading {p}: {e}")

    unique = _dedupe_decks(all_decks)

    # 验证卡 ID 在 STS2 真实存在
    if validate_cards:
        try:
            from networkV2.s1_schema.sim_catalog import GAME_CATALOG
            valid: list[dict] = []
            for d in unique:
                deck = d["deck"]
                kept = [c for c in deck if GAME_CATALOG.card_exists(c)]
                missing = set(deck) - set(kept)
                if missing:
                    logger.info(f"deck drop missing cards: {missing}")
                if len(kept) >= 8:
                    d["deck"] = kept
                    valid.append(d)
            unique = valid
        except Exception as e:
            logger.warning(f"Card validation skipped: {e}")

    logger.info(f"Loaded {len(unique)} unique real boss-ready decks from {len(all_decks)} samples")
    return unique


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    decks = load_real_boss_decks()
    print(f"\nLoaded {len(decks)} real boss-ready decks:")
    for i, d in enumerate(decks):
        dn = Counter(d["deck"])
        print(f"\n=== Deck {i+1}: {len(d['deck'])} cards @ floor {d['_floor']} {d['_state_type']} ===")
        for cid, n in dn.most_common():
            print(f"  {cid}: {n}")
        print(f"  relics: {d.get('relics', [])}")
