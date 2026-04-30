"""合成非战斗训练数据。

原因：rollout_full_run 走 HeadlessSim pipe proto 时，sim 对 combat 下 full_run_env/step
有拒绝 bug，无法端到端跑完整局收非战斗样本。暂走合成路径补齐 event / map / card_reward
/ campfire 几类 state_type 的 SFT 信号。

生成方式：按 state_type 手工构造一组"合理的"原始 state + legal_actions 组合，
喂给 non_combat_teacher 产出 chosen_index + reason，格式化成 messages JSONL。

不是最优数据，但**给模型一个有信号的非战斗老师**，避免它无脑选 idx=0。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

_STS2AI_ROOT = Path(__file__).resolve().parents[2]
if str(_STS2AI_ROOT) not in sys.path:
    sys.path.insert(0, str(_STS2AI_ROOT))

from llm.data_pipeline.non_combat_teacher import pick_non_combat
from llm.data_pipeline.state_renderer import render_state_text
from llm.paths import DATASETS_ROOT, ensure_dirs
from llm.prompts import load_system_prompt


# --- Map 节点类型（sim 的 label 字段实测值）---
_MAP_NODE_LABELS = ["monster", "elite", "event", "rest", "shop", "treasure"]

# --- Event 选项 label（NEOW / 普通事件常见模板）---
_EVENT_OPTION_LABELS = [
    # NEOW 开局
    ["NEOW.pages.INITIAL.options.NEW_LEAF.title",
     "NEOW.pages.INITIAL.options.STANDARD_DIFFICULTY.title"],
    # 好事件
    ["EVENT.GOLDEN_IDOL.TAKE", "EVENT.GOLDEN_IDOL.LEAVE"],
    ["EVENT.HEAL_FREE.HEAL_HP", "EVENT.HEAL_FREE.LEAVE"],
    ["EVENT.UPGRADE_RANDOM.UPGRADE", "EVENT.UPGRADE_RANDOM.LEAVE"],
    # 有代价的事件
    ["EVENT.SACRIFICE.LOSE_HP_GAIN_RELIC", "EVENT.SACRIFICE.LEAVE"],
    ["EVENT.CURSED_TOME.ADD_CURSE_GAIN_GOLD", "EVENT.CURSED_TOME.LEAVE"],
    # 中性
    ["EVENT.MYSTERY.OPTION_A", "EVENT.MYSTERY.OPTION_B", "EVENT.MYSTERY.LEAVE"],
]

# --- Card reward 组合（3 卡 + skip）---
_CARD_REWARD_POOLS = [
    ["BLUDGEON", "STRIKE_IRONCLAD", "SHRUG_IT_OFF"],
    ["INFLAME", "TWIN_STRIKE", "DEFEND_IRONCLAD"],
    ["DEMON_FORM", "POMMEL_STRIKE", "CLOTHESLINE"],
    ["FLEX", "ANGER", "STRIKE_IRONCLAD"],
    ["HEAVY_BLADE", "IRON_WAVE", "DEFEND_IRONCLAD"],
    ["METALLICIZE", "UPPERCUT", "TRUE_GRIT"],
    ["CINDER", "FORGOTTEN_RITUAL", "SETUP_STRIKE"],
    # 加入一些弱卡让模型学会跳
    ["STRIKE_IRONCLAD", "STRIKE_IRONCLAD", "DEFEND_IRONCLAD"],
    ["DEFEND_IRONCLAD", "DEFEND_IRONCLAD", "DEFEND_IRONCLAD"],
]

# --- Campfire 选项 ---
_CAMPFIRE_OPTIONS = [
    ["REST_SITE.REST", "REST_SITE.SMITH", "REST_SITE.LEAVE"],
]


def _make_legal(action_type: str, *, count: int | None = None, **kwargs) -> list[dict[str, Any]]:
    """构造一个 legal_action 列表。kwargs 会放到每个 action 里。"""
    if count is None:
        count = 1
    return [
        {
            "action": action_type,
            "type": action_type,
            "index": i,
            "card_index": kwargs.get("card_index_list", [None] * count)[i],
            "target_id": None,
            "col": kwargs.get("col_list", [None] * count)[i],
            "row": kwargs.get("row_list", [None] * count)[i],
            "slot": None,
            "is_enabled": True,
            "label": kwargs.get("label_list", [action_type] * count)[i],
        }
        for i in range(count)
    ]


def _base_state(state_type: str, *, hp: int = 60, max_hp: int = 80, gold: int = 100, floor: int = 5) -> dict[str, Any]:
    return {
        "state_type": state_type,
        "terminal": False,
        "run": {"character": "IRONCLAD", "act": 0, "floor_reached": floor, "floor": floor},
        "player": {"hp": hp, "max_hp": max_hp, "gold": gold, "character": "IRONCLAD"},
        "battle": {},
        "enemies": [],
    }


def generate_map_samples(rng: random.Random, n: int) -> list[tuple[dict, list[dict]]]:
    out = []
    for _ in range(n):
        # 2-4 个节点，类型随机
        n_nodes = rng.randint(2, 4)
        labels = [rng.choice(_MAP_NODE_LABELS) for _ in range(n_nodes)]
        hp = rng.randint(8, 80)
        floor = rng.randint(1, 15)
        state = _base_state("map", hp=hp, floor=floor, gold=rng.randint(0, 400))
        legal = _make_legal(
            "choose_map_node",
            count=n_nodes,
            col_list=[rng.randint(0, 6) for _ in range(n_nodes)],
            row_list=[floor] * n_nodes,
            label_list=labels,
        )
        out.append((state, legal))
    return out


def generate_event_samples(rng: random.Random, n: int) -> list[tuple[dict, list[dict]]]:
    out = []
    for _ in range(n):
        opts = rng.choice(_EVENT_OPTION_LABELS)
        hp = rng.randint(20, 80)
        state = _base_state("event", hp=hp, floor=rng.randint(1, 15))
        legal = _make_legal(
            "choose_event_option",
            count=len(opts),
            label_list=opts,
        )
        out.append((state, legal))
    return out


def generate_card_reward_samples(rng: random.Random, n: int) -> list[tuple[dict, list[dict]]]:
    out = []
    for _ in range(n):
        pool = rng.choice(_CARD_REWARD_POOLS)
        state = _base_state("card_reward", hp=rng.randint(30, 80))
        legal: list[dict[str, Any]] = []
        for i, cid in enumerate(pool):
            legal.append({
                "action": "select_card_reward",
                "type": "select_card_reward",
                "index": i,
                "card_index": i,
                "target_id": None,
                "col": None, "row": None, "slot": None,
                "is_enabled": True,
                "label": cid,
                "card_id": cid,
            })
        # 加 skip
        legal.append({
            "action": "skip_card_reward",
            "type": "skip_card_reward",
            "index": len(pool),
            "card_index": None, "target_id": None,
            "col": None, "row": None, "slot": None,
            "is_enabled": True,
            "label": "skip_card_reward",
        })
        out.append((state, legal))
    return out


def generate_campfire_samples(rng: random.Random, n: int) -> list[tuple[dict, list[dict]]]:
    out = []
    for _ in range(n):
        opts = rng.choice(_CAMPFIRE_OPTIONS)
        hp = rng.randint(10, 80)
        state = _base_state("campfire", hp=hp)
        legal = _make_legal(
            "choose_campfire_option",
            count=len(opts),
            label_list=opts,
        )
        out.append((state, legal))
    return out


def generate_proceed_samples(rng: random.Random, n: int) -> list[tuple[dict, list[dict]]]:
    out = []
    for _ in range(n):
        state_type = rng.choice(["post_combat", "event", "map"])
        state = _base_state(state_type)
        legal = _make_legal("proceed", count=1, label_list=["proceed"])
        out.append((state, legal))
    return out


def generate_claim_reward_samples(rng: random.Random, n: int) -> list[tuple[dict, list[dict]]]:
    out = []
    for _ in range(n):
        n_rewards = rng.randint(1, 3)
        state = _base_state("post_combat")
        types = rng.sample(["gold", "potion", "relic", "card"], n_rewards)
        legal = _make_legal(
            "claim_reward",
            count=n_rewards,
            label_list=types,
        )
        # 加 proceed 收尾
        legal.append({
            "action": "proceed", "type": "proceed", "index": n_rewards,
            "card_index": None, "target_id": None,
            "col": None, "row": None, "slot": None,
            "is_enabled": True, "label": "proceed",
        })
        out.append((state, legal))
    return out


def build_sample(state: dict, legal: list[dict], system_prompt: str) -> dict[str, Any] | None:
    dec = pick_non_combat(state, legal)
    if dec is None:
        return None
    user_msg = render_state_text(state, legal)
    assistant_msg = json.dumps(
        {"action_index": int(dec.action_index), "reason": dec.reason[:200]},
        ensure_ascii=False,
    )
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "meta": {
            "state_type": state["state_type"],
            "source": "synthetic_non_combat",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-type", type=int, default=80)
    parser.add_argument("--out-subdir", type=str, default="synthetic_non_combat_v0")
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    args = parser.parse_args()

    ensure_dirs()
    out_dir = DATASETS_ROOT / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    system_prompt = load_system_prompt()

    generators = [
        ("map", generate_map_samples),
        ("event", generate_event_samples),
        ("card_reward", generate_card_reward_samples),
        ("campfire", generate_campfire_samples),
        ("proceed", generate_proceed_samples),
        ("claim_reward", generate_claim_reward_samples),
    ]

    samples: list[dict] = []
    counts: dict[str, int] = {}
    for name, gen in generators:
        entries = gen(rng, args.n_per_type)
        added = 0
        for state, legal in entries:
            s = build_sample(state, legal, system_prompt)
            if s:
                samples.append(s)
                added += 1
        counts[name] = added
        print(f"[synth] {name:15s} → {added} samples")

    rng.shuffle(samples)
    eval_n = max(1, int(len(samples) * args.eval_ratio))
    eval_samples = samples[:eval_n]
    train_samples = samples[eval_n:]

    def _dump(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _dump(out_dir / "train.jsonl", train_samples)
    _dump(out_dir / "eval.jsonl", eval_samples)
    (out_dir / "meta.json").write_text(json.dumps({
        "total": len(samples),
        "train_size": len(train_samples),
        "eval_size": len(eval_samples),
        "by_type": counts,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[synth] total={len(samples)} train={len(train_samples)} eval={len(eval_samples)}")
    print(f"[synth] out: {out_dir}")
    print("\nsample example:")
    print(train_samples[0]["messages"][1]["content"])
    print("  assistant:", train_samples[0]["messages"][2]["content"])


if __name__ == "__main__":
    main()
