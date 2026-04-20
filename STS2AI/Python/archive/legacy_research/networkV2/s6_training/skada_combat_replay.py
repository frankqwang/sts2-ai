"""Skada replay combat:从 victory run 提取"玩家打过的每场战斗"作为训练 task。

核心洞察:
  胜利玩家的 run 里,每场战斗前的 (deck, relics, hp) 状态 都是"这时候这个难度下
  真实玩家的 build"。让 AI 用**同样 build** 打**同样 encounter**,天然解决了:
    1. deck-encounter 难度匹配(act1 boss 用 act1 deck 不用 act3 deck)
    2. curriculum(早期战斗简单 build,后期复杂 build)
    3. 真实分布(不是 hardcoded / synthetic deck)

  如果玩家赢了 → 说明**战术上有解**,AI 只需要学"用这个 build 怎么打这个敌人"。

Pipeline:
  1. iter_combat_tasks_from_run(rec):扫 skada run floor_timeline,
     对每个有 combat 的 floor 重建 pre-combat state
     → 产 list[{encounter_id, build, ref_floor, character, asc, run_id}]
  2. cotrainer 在 rollout 时抽 task,sim.reset(build=task.build, encounter=task.encounter)
  3. AI 打那场战斗,PPO 更新

用法:
    from networkV2.s6_training.skada_combat_replay import sample_combat_tasks
    tasks = sample_combat_tasks(fetcher, n_runs=50)   # 返 ~250 个 combat task
    for t in tasks:
        state = client.reset(encounter_id=t['encounter_id'], build=t['build'])
        # rollout...
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Iterator

from networkV2.s6_training.skada_state_rebuilder import (
    SkadaRunState, iter_timeline_with_state,
)


logger = logging.getLogger(__name__)


def _deck_to_sim_format(deck: list[str]) -> list[dict[str, Any]]:
    """skada lower_snake + `+` → sim UPPER_SNAKE + upgrade_level。"""
    out = []
    for cid in deck:
        s = str(cid or "").strip().lower()
        upg = 0
        while s.endswith("+"):
            s = s[:-1]
            upg += 1
        if s:
            out.append({"id": s.upper(), "upgrade_level": upg})
    return out


def _relics_to_sim_format(relics: list[str]) -> list[dict[str, Any]]:
    return [{"id": str(r).strip().upper()} for r in relics if r]


def iter_combat_tasks_from_run(rec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """扫一条 victory run,逐 combat 产 task(pre-combat state snapshot)。

    每个 task:
      encounter_id:  combat.encounter (sim 用)
      build:         { deck, relics, max_hp, current_hp, gold, max_energy }
      ref_floor:     此战的楼层
      room_type:     monster/elite/boss
      character:     run character
      ascension:     run asc
      run_id:        源 run id
    """
    run = rec.get("run", {}) or {}
    character = str(run.get("character", "")).upper()
    ascension = int(run.get("ascension", 0) or 0)
    run_id = int(run.get("run_id", 0) or 0)

    for floor_data, pre_state, post_state in iter_timeline_with_state(rec):
        combat = floor_data.get("combat")
        if not isinstance(combat, dict):
            continue
        encounter_id = str(combat.get("encounter", "") or "").strip()
        if not encounter_id:
            continue
        enc_type = str(combat.get("enc_type", "") or "").upper()
        room_type = {"M": "monster", "E": "elite", "B": "boss"}.get(enc_type, "monster")

        # pre_state 是"进入本 floor 前"的状态,也就是进战斗前(apply_floor_pre 走过 hp_before)
        deck_sim = _deck_to_sim_format(pre_state.deck)
        relics_sim = _relics_to_sim_format(pre_state.relics)
        if not deck_sim:
            continue

        task = {
            "encounter_id": encounter_id,
            "build": {
                "deck": deck_sim,
                "relics": relics_sim,
                "max_hp": pre_state.max_hp,
                "current_hp": max(pre_state.hp, 1),
                "gold": pre_state.gold,
                "max_energy": 3,
            },
            "ref_floor": pre_state.floor,
            "room_type": room_type,
            "character": character,
            "ascension": ascension,
            "run_id": run_id,
            "deck_size": len(deck_sim),
            "relic_count": len(relics_sim),
        }
        yield task


def stratified_sample_tasks(
    task_pool: list[dict[str, Any]],
    n_total: int,
    *,
    weights: dict[str, float] | None = None,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """按 room_type 加权抽样,保证 boss/elite 样本足够。

    默认权重:monster 0.3 / elite 0.35 / boss 0.35(pool 天然偏 monster,加权平衡)。
    """
    rng = rng or random.Random(0)
    weights = weights or {"monster": 0.3, "elite": 0.35, "boss": 0.35}

    by_room: dict[str, list[dict]] = {}
    for t in task_pool:
        by_room.setdefault(t["room_type"], []).append(t)
    total_w = sum(weights.get(r, 0) for r in by_room)
    if total_w <= 0:
        return rng.choices(task_pool, k=n_total) if task_pool else []
    out = []
    for rt, bucket in by_room.items():
        w = weights.get(rt, 0)
        if w <= 0 or not bucket:
            continue
        n = max(1, int(n_total * w / total_w))
        out.extend(rng.choices(bucket, k=n))
    # 如果凑不够,随机补
    while len(out) < n_total:
        out.append(rng.choice(task_pool))
    rng.shuffle(out)
    return out[:n_total]


def sample_combat_tasks(
    fetcher,
    *,
    n_runs: int = 50,
    room_types: set[str] | None = None,
    character: str | None = None,
    require_map_acts: bool = False,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """抽 N 条 victory run,提取所有 combat task,打平成列表。

    room_types 过滤:只保留 monster/elite/boss 其中一种或多种。
    return 可直接作为训练 batch 的 task 源。
    """
    rng = random.Random(seed)
    rows = fetcher.sample_clean_runs(
        n=n_runs, character=character, require_map_acts=require_map_acts,
    )
    all_tasks: list[dict[str, Any]] = []
    for row in rows:
        try:
            rec = fetcher.fetch_record(row)
        except Exception as e:
            logger.debug(f"fetch failed run_id={row.run_id}: {e}")
            continue
        for task in iter_combat_tasks_from_run(rec):
            if room_types and task["room_type"] not in room_types:
                continue
            all_tasks.append(task)
    rng.shuffle(all_tasks)
    logger.info(
        f"sampled {len(rows)} runs → {len(all_tasks)} combat tasks "
        f"({sum(1 for t in all_tasks if t['room_type']=='monster')} monster, "
        f"{sum(1 for t in all_tasks if t['room_type']=='elite')} elite, "
        f"{sum(1 for t in all_tasks if t['room_type']=='boss')} boss)"
    )
    return all_tasks


_MULTIPLAYER_CARD_PATTERNS = ("_MP_", "MULTIPLAYER_")


def _is_multiplayer_card(card_id: str) -> bool:
    """识别多人模式专用卡 (sim 单人训练不支持)。

    skada 数据里出现 e.g. STS2_AI_A_CARD_MULTIPLAYER_CARD_MP_WRIGGLE。
    pattern 匹配(不是名单):避免硬 list 漏卡。
    """
    u = card_id.upper()
    return any(p in u for p in _MULTIPLAYER_CARD_PATTERNS)


def iter_combat_chain_from_run(
    rec: dict[str, Any],
    *,
    supported_encounters: set[str] | None = None,
    supported_cards: set[str] | None = None,
    supported_relics: set[str] | None = None,
) -> list[dict[str, Any]]:
    """一条 run → 按 floor 顺序排列的完整 combat task list。

    **数据清洗** (skada v0.99-v0.103 混合 → sim v0.103 对齐):
      - supported_encounters: sim combat_catalog 白名单,缺失 → skip combat
        (根治 TOADPOLES_NORMAL not found / BATTLEWORN_DUMMY_EVENT_ENCOUNTER 等)
      - 多人模式 card (pattern `_MP_` / `MULTIPLAYER_`): **从 deck 移除**,不 skip task
        (skada 爬数据含多人 run,但单张 MP 卡不影响整个 deck 质量)
      - supported_cards (可选): sim 已认识的 card ID。**不用作 deck 过滤**
        (sim 用 generic `STRIKE` 名,skada 用 `STRIKE_REGENT` 后缀名; 命名约定差异
         不代表 sim 不支持该卡。留作未来做 fuzzy match 时再启用)

    白名单为 None 时不做对应过滤。
    """
    tasks = list(iter_combat_tasks_from_run(rec))
    tasks.sort(key=lambda t: t["ref_floor"])

    filtered: list[dict[str, Any]] = []
    for t in tasks:
        enc_id = t["encounter_id"].upper()
        # Filter 1: sim 不支持的 encounter (版本移除 / event / 错分类)
        if supported_encounters is not None and enc_id not in supported_encounters:
            continue
        # Filter 2: 清洗 deck
        #  (a) 去掉多人模式卡 (pattern _MP_/MULTIPLAYER_)
        #  (b) 去掉 sim 不支持的卡 (normalize 去角色后缀再和 sim generic card_id 比)
        orig_deck = t["build"]["deck"]
        clean_deck = []
        for c in orig_deck:
            cid = str(c.get("id", ""))
            if _is_multiplayer_card(cid):
                continue
            if supported_cards is not None:
                norm = _normalize_card_id(cid)
                if norm not in supported_cards:
                    continue  # sim 没这张卡 (版本移除等)
            clean_deck.append(c)
        if len(clean_deck) < 8:
            continue  # deck 剩太少不能打
        # Filter 3: 清洗 relics (sim 不支持的 mod relic 比如 EXTRARELICS-*)
        orig_relics = t["build"].get("relics", [])
        clean_relics = orig_relics
        if supported_relics is not None and orig_relics:
            clean_relics = [
                r for r in orig_relics
                if str(r.get("id", "")).strip().upper() in supported_relics
            ]
        if len(clean_deck) != len(orig_deck) or len(clean_relics) != len(orig_relics):
            t = dict(t)
            t["build"] = dict(t["build"])
            t["build"]["deck"] = clean_deck
            t["build"]["relics"] = clean_relics
            t["deck_size"] = len(clean_deck)
            t["relic_count"] = len(clean_relics)
        filtered.append(t)
    return filtered


def sample_combat_chains(
    fetcher,
    *,
    n_runs: int = 50,
    character: str | None = None,
    require_map_acts: bool = False,
    min_combats: int = 3,
    seed: int = 0,
    supported_encounters: set[str] | None = None,
    supported_cards: set[str] | None = None,
    supported_relics: set[str] | None = None,
) -> list[list[dict[str, Any]]]:
    """抽 N 条 victory run,每条产一个 floor-ordered chain。

    返回 list[list[task]],每个 sublist 是一个 run 的 combat 序列
    (按 floor 升序,从第一场战斗 → boss)。

    min_combats:过短的 chain (早期投降 run 等) 直接丢。典型 victory run 有 12-18 场。

    supported_{encounters,cards}:权威白名单(应来自 sim game_catalog),
    传入后会在构造 chain 时过滤掉 sim 不支持的 encounter/card
    (skada 数据版本 v0.102 vs sim v0.103 drift 根治)。
    """
    rng = random.Random(seed)
    rows = fetcher.sample_clean_runs(
        n=n_runs, character=character, require_map_acts=require_map_acts,
    )
    chains: list[list[dict[str, Any]]] = []
    n_filtered_empty = 0
    for row in rows:
        try:
            rec = fetcher.fetch_record(row)
        except Exception as e:
            logger.debug(f"fetch failed run_id={row.run_id}: {e}")
            continue
        chain = iter_combat_chain_from_run(
            rec,
            supported_encounters=supported_encounters,
            supported_cards=supported_cards,
            supported_relics=supported_relics,
        )
        if len(chain) < min_combats:
            n_filtered_empty += 1
            continue
        chains.append(chain)
    rng.shuffle(chains)
    lens = [len(c) for c in chains]
    logger.info(
        f"sampled {len(rows)} runs → {len(chains)} chains "
        f"(filtered {n_filtered_empty} too-short-after-validation, "
        f"avg {sum(lens)/max(1,len(lens)):.1f} combats/chain, "
        f"min={min(lens) if lens else 0}, max={max(lens) if lens else 0}, "
        f"total {sum(lens)} combats)"
    )
    return chains


# 5 个 character id 对应 skada card 的后缀 (skada `STRIKE_DEFECT` vs sim `STRIKE`)
_CHARACTER_SUFFIXES: tuple[str, ...] = (
    "_IRONCLAD", "_SILENT", "_DEFECT", "_REGENT", "_NECROBINDER", "_WATCHER",
)


def _normalize_card_id(card_id: str) -> str:
    """skada 带角色后缀 → sim generic 名。

    例:
      STRIKE_DEFECT → STRIKE
      ELECTRODYNAMICS → ELECTRODYNAMICS (无变化)
    """
    u = str(card_id).upper()
    for suf in _CHARACTER_SUFFIXES:
        if u.endswith(suf):
            return u[: -len(suf)]
    return u


@dataclass
class SimSupported:
    """sim 权威白名单集合(所有 id 大写)。"""
    encounters: set[str]
    cards: set[str]
    relics: set[str]
    potions: set[str]


def load_sim_supported_lists(client) -> SimSupported:
    """从 sim `game_catalog` + `combat_catalog` API 拿权威白名单。

    cards/relics/potions 是 generic ID 大写集合
    (skada 带角色后缀的 ID 要 normalize 再查)。

    **2026-04-19**:proto wire (CombatSession) 不支持 catalog opcode,
    sim 侧调用会 NotImplementedError。fallback 到 `data/source_knowledge.sqlite`
    拿白名单 — sqlite 由 sim 导出保证内容权威。任一路径失败保证返回非空 set,
    避免 `supported_xxx=None` 导致 skada 清洗完全失效。
    """
    encs: set[str] = set()
    cards: set[str] = set()
    relics: set[str] = set()
    potions: set[str] = set()
    try:
        cc = client._call("combat_catalog") if hasattr(client, "_call") else client.call("combat_catalog")
        for item in (cc.get("encounters") or []):
            if isinstance(item, dict):
                eid = str(item.get("encounter_id") or item.get("encounterId") or "").strip().upper()
                if eid:
                    encs.add(eid)
    except Exception as e:
        logger.warning(f"combat_catalog failed: {e}")
    try:
        gc = client._call("game_catalog") if hasattr(client, "_call") else client.call("game_catalog")
        for item in (gc.get("cards") or []):
            if isinstance(item, dict):
                cid = str(item.get("card_id") or item.get("id") or "").strip().upper()
                if cid:
                    cards.add(cid)
            elif isinstance(item, str):
                cards.add(item.strip().upper())
        for item in (gc.get("relics") or []):
            if isinstance(item, dict):
                rid = str(item.get("relic_id") or item.get("id") or "").strip().upper()
                if rid:
                    relics.add(rid)
            elif isinstance(item, str):
                relics.add(item.strip().upper())
        for item in (gc.get("potions") or []):
            if isinstance(item, dict):
                pid = str(item.get("potion_id") or item.get("id") or "").strip().upper()
                if pid:
                    potions.add(pid)
            elif isinstance(item, str):
                potions.add(item.strip().upper())
    except Exception as e:
        logger.warning(f"game_catalog failed: {e}")

    # Fallback 到 sqlite(proto wire / sim catalog 不可用时)
    if not encs or not cards or not relics or not potions:
        try:
            _encs, _cards, _relics, _potions = _load_sqlite_whitelists()
            if not encs:
                encs = _encs
            if not cards:
                cards = _cards
            if not relics:
                relics = _relics
            if not potions:
                potions = _potions
            logger.info("filled missing whitelists from sqlite fallback")
        except Exception as e:
            logger.warning(f"sqlite fallback failed: {e}")

    logger.info(
        f"sim supported: {len(encs)} encounters, {len(cards)} cards, "
        f"{len(relics)} relics, {len(potions)} potions"
    )
    return SimSupported(encounters=encs, cards=cards, relics=relics, potions=potions)


def _load_sqlite_whitelists() -> tuple[set[str], set[str], set[str], set[str]]:
    """从 data/source_knowledge.sqlite 读取 encounters/cards/relics/potions 列表。

    sqlite 由 sim ModelDb 导出(`data/export_game_catalog_runtime.py`),
    是 sim 权威内容的静态快照。proto wire 下 CombatSession 无法访问 sim
    game_catalog 时用这份 fallback。

    **过滤 `*_EVENT_ENCOUNTER`**:这类 encounter(BattlewornDummyEvent /
    DenseVegetation / FakeMerchant / MysteriousKnight / PunchOff /
    TheArchitect 等)依赖 event 上下文传入 `DummySetting` 之类的配置,
    独立 combat_reset 会抛 "Setting must be set!"。不适合训练。
    """
    import sqlite3
    from pathlib import Path as _Path
    db_path = _Path(__file__).resolve().parents[2] / "data" / "source_knowledge.sqlite"
    if not db_path.exists():
        return set(), set(), set(), set()
    con = sqlite3.connect(str(db_path))
    try:
        def _fetch(table: str) -> set[str]:
            try:
                return {
                    str(r[0]).strip().upper()
                    for r in con.execute(f"SELECT id FROM {table}").fetchall()
                    if r and r[0]
                }
            except Exception:
                return set()
        encs = _fetch("encounters")
        encs = {e for e in encs if not e.endswith("_EVENT_ENCOUNTER")}
        return (
            encs,
            _fetch("cards"),
            _fetch("relics"),
            _fetch("potions"),
        )
    finally:
        con.close()


def chain_stats(chains: list[list[dict[str, Any]]]) -> dict[str, Any]:
    from collections import Counter
    if not chains:
        return {}
    room_by_pos = Counter()
    lens = []
    all_enc = Counter()
    for chain in chains:
        lens.append(len(chain))
        for t in chain:
            room_by_pos[t["room_type"]] += 1
            all_enc[t["encounter_id"]] += 1
    return {
        "n_chains": len(chains),
        "avg_len": sum(lens) / len(lens),
        "min_len": min(lens),
        "max_len": max(lens),
        "total_combats": sum(lens),
        "by_room_type": dict(room_by_pos),
        "top_encounters": all_enc.most_common(10),
    }


def task_stats(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """诊断信息。"""
    if not tasks:
        return {}
    from collections import Counter
    by_enc = Counter(t["encounter_id"] for t in tasks)
    by_room = Counter(t["room_type"] for t in tasks)
    by_floor_bucket = Counter()
    by_deck_size = Counter()
    for t in tasks:
        f = t["ref_floor"]
        b = "act1" if f <= 17 else "act2" if f <= 34 else "act3"
        by_floor_bucket[b] += 1
        by_deck_size[min(t["deck_size"] // 3 * 3, 30)] += 1
    return {
        "total": len(tasks),
        "by_room_type": dict(by_room),
        "by_act": dict(by_floor_bucket),
        "by_deck_size_bucket": dict(sorted(by_deck_size.items())),
        "top_encounters": by_enc.most_common(10),
    }
