"""Build a tiny planner-hint seed dataset from local guide knowledge.

This is for smoke training and prompt validation, not a substitute for teacher
labels. The examples are deliberately small, English-only, and use the same
planner prompt renderer as live inference so retrieved_knowledge is present in
the user message.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from llm.data_pipeline.planner_hint import render_planner_hint_user_message  # noqa: E402
from llm.paths import DATASETS_ROOT, ensure_dirs  # noqa: E402
from llm.prompts import load_system_prompt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DATASETS_ROOT / f"planner_hint_seed_{datetime.now().strftime('%Y%m%d-%H%M%S')}"))
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _base_player(deck: list[str], relics: list[str] | None = None, potions: list[str] | None = None) -> dict[str, Any]:
    return {
        "character": "IRONCLAD",
        "hp": 62,
        "max_hp": 80,
        "gold": 99,
        "deck": [{"id": card_id} for card_id in deck],
        "relics": [{"id": relic_id} for relic_id in (relics or ["BURNING_BLOOD"])],
        "potions": [{"id": potion_id} for potion_id in (potions or [])],
    }


def _card(card_id: str, cost: int, desc: str, damage: dict[str, int] | None = None, ctype: str = "attack") -> dict[str, Any]:
    out: dict[str, Any] = {"id": card_id, "cost": cost, "type": ctype, "description": desc}
    if damage:
        out["preview_damage_per_target"] = damage
    return out


def _enemy(target_id: int, monster_id: str, hp: int, intent: str, damage: int = 0, block: int = 0) -> dict[str, Any]:
    return {
        "target_id": target_id,
        "monster_id": monster_id,
        "hp": hp,
        "max_hp": max(hp, 1),
        "block": block,
        "intent_type": intent,
        "intent_damage": damage,
        "intent_hits": 1,
        "is_alive": True,
    }


def _state(encounter_id: str, floor: int, hand: list[dict[str, Any]], enemies: list[dict[str, Any]], *, deck: list[str], relics: list[str] | None = None, potions: list[str] | None = None) -> dict[str, Any]:
    return {
        "run": {"act": 1, "floor": floor, "gold": 99},
        "player": _base_player(deck, relics=relics, potions=potions),
        "battle": {
            "encounter_id": encounter_id,
            "round_number_raw": 1,
            "energy": 3,
            "player": {"block": 0},
            "hand": hand,
        },
        "enemies": enemies,
    }


def _examples() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    starter = ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH"]
    return [
        (
            _state(
                "CULTISTS_NORMAL",
                4,
                [
                    _card("STRIKE_IRONCLAD", 1, "Deal 6 damage.", {"1": 6, "2": 6}),
                    _card("DEFEND_IRONCLAD", 1, "Gain 5 Block.", ctype="skill"),
                    _card("BASH", 2, "Deal 8 damage. Apply 2 Vulnerable.", {"1": 8, "2": 8}),
                ],
                [
                    _enemy(1, "CALCIFIED_CULTIST", 39, "Buff"),
                    _enemy(2, "DAMP_CULTIST", 51, "Buff"),
                ],
                deck=starter,
            ),
            {
                "battle_objective": "Use the first safe turns to set a Vulnerable damage window before Cultist scaling matters.",
                "enemy_focus": "Focus one CULTIST at a time; prefer DAMP_CULTIST when both are equally reachable, otherwise finish the lower-risk kill.",
                "deck_usage": "BASH is valuable when STRIKE_IRONCLAD follow-up can exploit Vulnerable this turn or next turn.",
                "risk_tradeoff": "Accept small HP loss for tempo early, but switch to block when attacks become meaningful.",
                "resource_timing": "Spend 2 energy on BASH only when the target choice and next attacks can use the debuff.",
                "potion_stance": "Save potions unless scaling creates a large unavoidable HP swing.",
                "kill_order": ["DAMP_CULTIST", "CALCIFIED_CULTIST"],
                "danger_notes": ["Do not split damage so both Cultists keep scaling."],
            },
        ),
        (
            _state(
                "SLIMES_NORMAL",
                3,
                [
                    _card("STRIKE_IRONCLAD", 1, "Deal 6 damage.", {"1": 6, "2": 6, "3": 6}),
                    _card("STRIKE_IRONCLAD", 1, "Deal 6 damage.", {"1": 6, "2": 6, "3": 6}),
                    _card("DEFEND_IRONCLAD", 1, "Gain 5 Block.", ctype="skill"),
                ],
                [
                    _enemy(1, "TWIG_SLIME_M", 26, "StatusCard"),
                    _enemy(2, "LEAF_SLIME_S", 10, "Attack", 4),
                    _enemy(3, "TWIG_SLIME_S", 8, "Attack", 4),
                ],
                deck=starter,
            ),
            {
                "battle_objective": "Reduce enemy count quickly instead of spreading low damage across the Slime group.",
                "enemy_focus": "Prefer killing small attackers when reachable, then focus a medium Slime.",
                "deck_usage": "Starter attacks are enough to remove low-HP Slimes; keep DEFEND for turns where attacks cannot reduce incoming enough.",
                "risk_tradeoff": "Small damage is acceptable if it removes an attacker and prevents future status pressure.",
                "resource_timing": "Use cheap attacks to convert kills before spending energy on low-impact block.",
                "potion_stance": "Do not spend potions in a normal Slime fight unless HP would swing badly.",
                "kill_order": ["enemy3", "enemy2", "enemy1"],
                "danger_notes": ["Status pressure can make slow cleanup worse."],
            },
        ),
        (
            _state(
                "CHOMPERS_NORMAL",
                8,
                [
                    _card("BASH", 2, "Deal 8 damage. Apply 2 Vulnerable.", {"1": 8}),
                    _card("DEFEND_IRONCLAD", 1, "Gain 5 Block.", ctype="skill"),
                    _card("DEFEND_IRONCLAD", 1, "Gain 5 Block.", ctype="skill"),
                ],
                [_enemy(1, "CHOMPER", 32, "Attack", 12)],
                deck=starter,
            ),
            {
                "battle_objective": "Balance HP preservation with a Vulnerable setup so the fight does not drag.",
                "enemy_focus": "Single-target fight; keep pressure on CHOMPER while respecting attack turns.",
                "deck_usage": "BASH can set up future damage, but current DEFEND cards may be needed if the attack is the main risk.",
                "risk_tradeoff": "Block more when unblocked damage is meaningful and no immediate kill line exists.",
                "resource_timing": "Choose BASH on safer turns or when follow-up attacks are likely soon.",
                "potion_stance": "Save potions unless the current attack creates a large HP loss.",
                "kill_order": ["enemy1"],
                "danger_notes": ["Do not let setup ignore a heavy current attack."],
            },
        ),
        (
            _state(
                "BOWLBUGS_NORMAL",
                9,
                [
                    _card("POMMEL_STRIKE", 1, "Deal 8 damage. Draw 1 card.", {"1": 8, "2": 8}),
                    _card("STRIKE_IRONCLAD", 1, "Deal 6 damage.", {"1": 6, "2": 6}),
                    _card("BASH", 2, "Deal 8 damage. Apply 2 Vulnerable.", {"1": 8, "2": 8}),
                ],
                [
                    _enemy(1, "BOWLBUG", 8, "Attack", 5),
                    _enemy(2, "BOWLBUG", 24, "Attack", 5),
                ],
                deck=starter + ["POMMEL_STRIKE"],
            ),
            {
                "battle_objective": "Remove one attacker with efficient damage, then use Vulnerable to finish the remaining Bowlbug.",
                "enemy_focus": "Kill the low-HP attacker first when POMMEL_STRIKE or STRIKE can do it cleanly.",
                "deck_usage": "POMMEL_STRIKE is strong when it kills and draws into more options instead of overkilling with BASH.",
                "risk_tradeoff": "Reducing enemy count can be better defense than blocking small split attacks.",
                "resource_timing": "Prefer cheap lethal plus draw before spending 2 energy on BASH.",
                "potion_stance": "Normal fight; hold potions unless a bad draw creates major HP risk.",
                "kill_order": ["enemy1", "enemy2"],
                "danger_notes": ["Do not overkill a low-HP target with the expensive setup card."],
            },
        ),
        (
            _state(
                "HAND_DRILL_BLOCK_TEST",
                11,
                [
                    _card("STRIKE_IRONCLAD", 1, "Deal 6 damage.", {"1": 6}),
                    _card("POMMEL_STRIKE", 1, "Deal 8 damage. Draw 1 card.", {"1": 8}),
                    _card("BASH", 2, "Deal 8 damage. Apply 2 Vulnerable.", {"1": 8}),
                ],
                [_enemy(1, "BLOCKING_ENEMY", 24, "Attack", 7, block=5)],
                deck=starter + ["POMMEL_STRIKE"],
                relics=["BURNING_BLOOD", "HAND_DRILL"],
            ),
            {
                "battle_objective": "Use HAND_DRILL block-break timing to create a Vulnerable payoff window.",
                "enemy_focus": "Single target; break Block before the largest follow-up attack when feasible.",
                "deck_usage": "Cheap attacks can trigger HAND_DRILL before BASH or POMMEL_STRIKE payoff damage.",
                "risk_tradeoff": "Accept modest damage only if block-break plus Vulnerable meaningfully shortens the fight.",
                "resource_timing": "Do not spend the payoff attack before the Block break if another card can trigger HAND_DRILL first.",
                "potion_stance": "Hold potions unless the Block-break line cannot prevent a large hit.",
                "kill_order": ["enemy1"],
                "danger_notes": ["HAND_DRILL needs an actual Block break to matter."],
            },
        ),
        (
            _state(
                "SOUL_FYSH",
                16,
                [
                    _card("BASH", 2, "Deal 8 damage. Apply 2 Vulnerable.", {"1": 8}),
                    _card("STRIKE_IRONCLAD", 1, "Deal 6 damage.", {"1": 6}),
                    _card("DEFEND_IRONCLAD", 1, "Gain 5 Block.", ctype="skill"),
                ],
                [_enemy(1, "SOUL_FYSH", 110, "Attack", 12)],
                deck=starter,
                potions=["FORTIFIER"],
            ),
            {
                "battle_objective": "Build steady damage while preparing for Soul Fysh pattern swings.",
                "enemy_focus": "Single boss target; time Vulnerable and attacks around safer damage windows.",
                "deck_usage": "BASH can start a damage window, but the deck must still preserve HP through fixed attack turns.",
                "risk_tradeoff": "Do not trade large HP into boss pattern pressure unless it creates a decisive damage lead.",
                "resource_timing": "Use setup before strong follow-up turns; avoid wasting Vulnerable during low-output hands.",
                "potion_stance": "FORTIFIER is worth using when added Block prevents a major boss hit.",
                "kill_order": ["enemy1"],
                "danger_notes": ["Boss pattern knowledge matters more than normal-fight tempo."],
            },
        ),
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    ensure_dirs()
    system_prompt = load_system_prompt("planner_hint")
    rows = []
    for index, (state, assistant) in enumerate(_examples()):
        rows.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": render_planner_hint_user_message(state)},
                {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False, separators=(",", ":"))},
            ],
            "meta": {
                "source": "planner_hint_seed_rag",
                "example_index": index,
                "encounter_id": state.get("battle", {}).get("encounter_id"),
            },
        })

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    eval_n = int(round(len(rows) * max(0.0, min(0.9, args.eval_ratio))))
    eval_rows = rows[:eval_n]
    train_rows = rows[eval_n:]

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "eval.jsonl", eval_rows)
    (out_dir / "summary.json").write_text(
        json.dumps({
            "kind": "planner_hint_seed_rag_dataset",
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "rows": len(rows),
            "train": len(train_rows),
            "eval": len(eval_rows),
            "uses_retrieved_knowledge": True,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
