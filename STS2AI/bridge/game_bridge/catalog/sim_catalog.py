"""统一游戏数据 catalog 接口。

**规范**（SCHEMA_CONVENTION.md）：所有涉及游戏 id / 命名的数据，
必须通过本模块查询，不得在业务代码里硬编码。

数据源优先级：
  1. Sim API（combat_catalog、get_state 等）—— 最实时，跟随游戏代码版本
  2. source_knowledge.sqlite —— build_source_database.py 从 C# 代码提取的 snapshot
  3. （未来）专用 catalog API：cards_catalog / relics_catalog / monsters_catalog

Python 侧统一入口：
  from game_bridge.catalog.sim_catalog import GAME_CATALOG
  GAME_CATALOG.encounters(room_type='boss')           # list of encounter dict
  GAME_CATALOG.monster_powers('frog_knight')          # ['PlatingPower', 'StrengthPower', 'FrailPower']
  GAME_CATALOG.encounter_monster_ids('doormaker_boss') # ['DOORMAKER', 'DOOR']

如果运行时 sim client 可用，会优先用 API；否则 fallback sqlite。
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any


_NEW_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "game_wiki" / "game_catalog.sqlite"
_OLD_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "source_knowledge.sqlite"
_DB_PATH = _NEW_DB_PATH if _NEW_DB_PATH.exists() else _OLD_DB_PATH


def _pascal_to_snake(s: str) -> str:
    """FrogKnight → frog_knight；BattlewornDummyTimeLimit → battleworn_dummy_time_limit"""
    if not s:
        return s
    out = []
    for i, ch in enumerate(s):
        if ch.isupper() and i > 0:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


class GameCatalog:
    """游戏数据统一访问点。

    优先查 sim API（如果 attach 了 client），fallback 到 sqlite。
    sqlite 由 tools/python/data/build_source_database.py 从 C# ModelDb 提取。
    """

    def __init__(self, db_path: Path = _DB_PATH):
        self._db_path = db_path
        self._sim_client = None      # 可选：attach live sim 时绕 sqlite
        self._sim_catalog_cache: dict | None = None
        self._game_catalog_cache: dict | None = None  # game_catalog API 缓存
        # Power 元数据（按 class_name 索引）：base_classes + is_debuff_hint
        # 由 attach_sim 填充，token_bank_builder 用来精确判定 semantic group / debuff
        self._power_metadata_by_class: dict[str, dict] = {}

    def attach_sim(self, client: Any) -> None:
        """把 game bridge sim client 接进来，后续所有查询走 sim API（game_catalog）。

        兼容多种 client 类型：
          - PipeBackedCombatTrainingClient：有 `_call(method, params)` 方法
          - BinaryBackedFullRunClient：没有 `_call`，底层有 `_pipe.call(...)`
          - 其他自定义 client：只要有 `_pipe` 且是 PipeClient 即可
        """
        self._sim_client = client
        self._sim_catalog_cache = None
        self._game_catalog_cache = None
        self._power_metadata_by_class = {}
        # 启动时预取 game_catalog，后续查询都从 cache
        import logging as _logging
        _log = _logging.getLogger(__name__)
        try:
            self._game_catalog_cache = self._invoke_sim(client, "game_catalog")
            cache = self._game_catalog_cache or {}
            _log.info(f"[GAME_CATALOG] game_catalog response keys: {list(cache.keys())}")
            powers_raw = cache.get("powers")
            _log.info(f"[GAME_CATALOG] powers type={type(powers_raw).__name__} len={len(powers_raw) if powers_raw else 0}")
            if powers_raw:
                _log.info(f"[GAME_CATALOG] first power sample: {powers_raw[0]}")
            # 提取 powers 元数据到 dict[class_name, {base_classes, is_debuff_hint}]
            for p in powers_raw or []:
                cls = p.get("class_name")
                if cls:
                    self._power_metadata_by_class[str(cls)] = {
                        "base_classes": list(p.get("base_classes") or []),
                        "is_debuff_hint": bool(p.get("is_debuff_hint", False)),
                    }
        except Exception as e:
            _log.warning(f"[GAME_CATALOG] attach_sim failed: {type(e).__name__}: {e}")

    @staticmethod
    def _invoke_sim(client: Any, method: str, params: dict | None = None) -> dict:
        """调 sim RPC，兼容多种 client 接口。"""
        # 路径 1：client 自带 _call (PipeBackedCombatTrainingClient)
        call_fn = getattr(client, "_call", None)
        if callable(call_fn):
            return call_fn(method, params)
        # 路径 2：BinaryBackedFullRunClient 底层 _pipe.call
        pipe = getattr(client, "_pipe", None)
        if pipe is not None and hasattr(pipe, "call"):
            # 有些 client 的 _pipe 需要先 ensure_connected
            ensure = getattr(client, "_ensure_connected", None)
            if callable(ensure):
                ensure()
            return pipe.call(method, params)
        raise AttributeError(f"sim client {type(client).__name__} has no _call or _pipe.call")

    def _connect(self):
        if not self._db_path.exists():
            return None
        return sqlite3.connect(str(self._db_path))

    # ------------------------------------------------------------------
    # Encounters
    # ------------------------------------------------------------------

    def encounters(self, room_type: str | None = None) -> list[dict[str, Any]]:
        """返回所有 encounters。优先 game_catalog API（含 act_index），
        次选 combat_catalog API（仅 id+room_type），最后 sqlite fallback。

        每个 entry: {encounter_id, room_type, monster_ids?, act_index?}
        """
        # 1. game_catalog 缓存（最全：id/room_type/monster_ids/act_index）
        if self._game_catalog_cache is not None:
            encs = self._game_catalog_cache.get("encounters") or []
            if room_type:
                encs = [e for e in encs if e.get("room_type", "").lower() == room_type.lower()]
            return list(encs)

        # 2. combat_catalog fallback（旧版 sim 可能无 game_catalog）
        if self._sim_client is not None:
            if self._sim_catalog_cache is None:
                try:
                    cat = self._sim_client.combat_catalog()
                    self._sim_catalog_cache = cat
                except Exception:
                    self._sim_catalog_cache = None
            if self._sim_catalog_cache:
                encs = self._sim_catalog_cache.get("encounters") or []
                if room_type:
                    encs = [e for e in encs if e.get("room_type", "").lower() == room_type.lower()]
                return list(encs)

        # 2. sqlite fallback
        conn = self._connect()
        if conn is None:
            return []
        try:
            cur = conn.cursor()
            if room_type:
                rows = cur.execute(
                    "SELECT id, room_type FROM encounters WHERE LOWER(room_type)=? ORDER BY id",
                    (room_type.lower(),),
                ).fetchall()
            else:
                rows = cur.execute(
                    "SELECT id, room_type FROM encounters ORDER BY room_type, id"
                ).fetchall()
            return [
                {"encounter_id": eid, "room_type": (rt or "").lower()}
                for eid, rt in rows
            ]
        finally:
            conn.close()

    def encounter_monsters(self, encounter_id: str) -> list[str]:
        """返回 encounter 包含的 monster class name 列表。优先 sim API。"""
        # 1. Sim game_catalog API（如果已 attach + cache）
        if self._game_catalog_cache is not None:
            for enc in self._game_catalog_cache.get("encounters", []):
                if enc.get("encounter_id", "").lower() == encounter_id.lower():
                    return list(enc.get("monster_ids", []))
            return []

        # 2. sqlite fallback
        conn = self._connect()
        if conn is None:
            return []
        try:
            cur = conn.cursor()
            row = cur.execute(
                "SELECT monster_ids_json FROM encounters WHERE id=? OR LOWER(id)=?",
                (encounter_id, encounter_id.lower()),
            ).fetchone()
            if not row or not row[0]:
                return []
            try:
                data = json.loads(row[0])
                if isinstance(data, list):
                    out = []
                    for item in data:
                        if isinstance(item, str):
                            out.append(item)
                        elif isinstance(item, dict):
                            out.append(str(item.get("id") or item.get("monster_id") or ""))
                    return [x for x in out if x]
            except Exception:
                return []
        finally:
            conn.close()
        return []

    # ------------------------------------------------------------------
    # Monsters
    # ------------------------------------------------------------------

    def monster_powers(self, monster_id: str) -> list[str]:
        """返回 monster 的初始 power class 名列表（来自 C# ModelDb）。

        注：sqlite 里 monster id 是 snake_case（`frog_knight`），但
        encounters.possible_monsters_json 里引用的是 PascalCase（`FrogKnight`）。
        做 casing-agnostic 匹配。
        """
        conn = self._connect()
        if conn is None:
            return []
        try:
            cur = conn.cursor()
            # PascalCase → snake_case 转换
            # FrogKnight → frog_knight
            snake = _pascal_to_snake(monster_id) if monster_id and monster_id[0].isupper() else monster_id
            row = cur.execute(
                "SELECT powers_json FROM monsters WHERE id=? OR LOWER(id)=?",
                (snake, snake.lower()),
            ).fetchone()
            if not row or not row[0]:
                return []
            try:
                return list(json.loads(row[0]))
            except Exception:
                return []
        finally:
            conn.close()

    def encounter_has_power(self, encounter_id: str, power_class: str) -> bool:
        """encounter 中任意 monster 是否持有该 power class。"""
        for mid in self.encounter_monsters(encounter_id):
            if power_class in self.monster_powers(mid):
                return True
        return False

    def encounter_difficulty_signals(self, encounter_id: str) -> dict[str, Any]:
        """从 DB 派生 encounter 难度指纹（用于 auto curriculum）。

        注意：仅 DB-level 数据；运行时实际 hp 可能因 ascension 变化。

        分级规则：
          has_block_mechanic: 持续产生 block 的机制（对新手 deck 致命，必须 buffed）
          has_hard_scaling:   快速成长机制（RitualPower 等），mid-term 威胁
          is_starter_blocker: 任一上面为 True → 新手不宜
        """
        monsters = self.encounter_monsters(encounter_id)
        all_powers: list[str] = []
        for mid in monsters:
            all_powers.extend(self.monster_powers(mid))
        power_set = set(all_powers)
        # 真正让 starter deck 打不穿的 block 机制
        has_block = any(p in power_set for p in {
            "PlatingPower",        # 回合结束加 block
            "BarricadePower",      # block 不清零
            "IntangiblePower",     # 伤害降到 1
            "HardenedShellPower",  # 首次受击变 block
            "SlipperyPower",       # 受击减伤
            "FlightPower",         # 多次受击才伤 HP
        })
        # 明确 "随回合快速成长" 的机制（不含单纯 StrengthPower，那只是初始值）
        has_hard_scaling = any(p in power_set for p in {
            "RitualPower",         # 每回合 +strength
            "EnragePower",         # 打 skill 时 +strength
            "PlowPower",           # 未测，保守算 scaling
            "CrabRagePower",       # 特殊 rage scaling
        })
        return {
            "encounter_id": encounter_id,
            "monster_count": len(monsters),
            "monster_ids": monsters,
            "power_set": sorted(power_set),
            "has_block_mechanic": has_block,
            "has_hard_scaling": has_hard_scaling,
            "has_minion_spawn": "MinionPower" in power_set,
            "is_starter_blocker": has_block,  # 主要看 block，scaling 问题 dense shaping 能 partial 解
        }

    # ------------------------------------------------------------------
    # Power class metadata (sim game_catalog → Python 直接访问)
    # ------------------------------------------------------------------

    def power_base_classes(self, class_name: str) -> list[str]:
        """返回 power class 的继承链（如 ["TriggerOnAttackedPower", "PowerModel"]）。

        由 attach_sim 从 game_catalog API 预取并 cache；若未 attach sim，返回空列表
        （调用方应 fallback 到 heuristic）。
        """
        meta = self._power_metadata_by_class.get(class_name)
        if meta is None:
            return []
        return list(meta.get("base_classes") or [])

    def power_is_debuff_hint(self, class_name: str) -> bool | None:
        """返回 power class 的 debuff 判断（来自 sim 的 heuristic）。

        未挂 sim 或未知 class → None（调用方应 fallback 到本地 heuristic）。
        """
        meta = self._power_metadata_by_class.get(class_name)
        if meta is None:
            return None
        return bool(meta.get("is_debuff_hint", False))

    def power_metadata(self, class_name: str) -> dict | None:
        """返回完整 power 元数据 dict {base_classes, is_debuff_hint}；未知返回 None。"""
        return self._power_metadata_by_class.get(class_name)

    # ------------------------------------------------------------------
    # Power class vocab
    # ------------------------------------------------------------------

    @lru_cache(maxsize=1)
    def monster_power_vocab(self) -> list[str]:
        """所有 monster 用过的 power class，按频次倒排。"""
        conn = self._connect()
        if conn is None:
            return []
        try:
            cur = conn.cursor()
            cnt = Counter()
            for (pj,) in cur.execute(
                "SELECT powers_json FROM monsters WHERE powers_json IS NOT NULL"
            ):
                try:
                    for p in json.loads(pj):
                        cnt[p] += 1
                except Exception:
                    pass
            return [p for p, _ in cnt.most_common()]
        finally:
            conn.close()

    @lru_cache(maxsize=1)
    def player_power_vocab(self) -> list[str]:
        """出现在 card + relic powers_json 里的 class（玩家能获得的 power）。"""
        conn = self._connect()
        if conn is None:
            return []
        try:
            cur = conn.cursor()
            cnt = Counter()
            for table in ("cards", "relics"):
                for (pj,) in cur.execute(
                    f"SELECT powers_json FROM {table} WHERE powers_json IS NOT NULL"
                ):
                    try:
                        for p in json.loads(pj):
                            cnt[p] += 1
                    except Exception:
                        pass
            return [p for p, _ in cnt.most_common()]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Cards
    # ------------------------------------------------------------------

    def card_commands(self, card_id: str) -> list[str]:
        """返回 card 的 commands_json（动作 verb 列表，如 [Attack, Block, Draw]）。"""
        conn = self._connect()
        if conn is None:
            return []
        try:
            cur = conn.cursor()
            row = cur.execute(
                "SELECT commands_json FROM cards WHERE id=? OR LOWER(id)=?",
                (card_id, card_id.lower()),
            ).fetchone()
            if not row or not row[0]:
                return []
            try:
                return list(json.loads(row[0]))
            except Exception:
                return []
        finally:
            conn.close()

    def card_exists(self, card_id: str) -> bool:
        """检查 card_id 是否在真实游戏数据里存在（大小写不敏感）。"""
        conn = self._connect()
        if conn is None:
            return False
        try:
            cur = conn.cursor()
            row = cur.execute(
                "SELECT 1 FROM cards WHERE id=? OR LOWER(id)=? LIMIT 1",
                (card_id, card_id.lower()),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    @lru_cache(maxsize=1)
    def draw_cards(self) -> frozenset[str]:
        """所有 commands_json 含 'Draw' 的卡 ID（uppercase + underscore）。"""
        conn = self._connect()
        if conn is None:
            return frozenset()
        try:
            cur = conn.cursor()
            ids: set[str] = set()
            for (cid, cj) in cur.execute(
                "SELECT id, commands_json FROM cards WHERE commands_json LIKE '%Draw%'"
            ):
                try:
                    if "Draw" in json.loads(cj):
                        ids.add(cid.upper())
                except Exception:
                    pass
            return frozenset(ids)
        finally:
            conn.close()

    @lru_cache(maxsize=1)
    def heal_cards(self) -> frozenset[str]:
        """所有 commands_json 含 'Heal' / 'Restore' 的卡。"""
        conn = self._connect()
        if conn is None:
            return frozenset()
        try:
            cur = conn.cursor()
            ids: set[str] = set()
            for (cid, cj) in cur.execute(
                "SELECT id, commands_json FROM cards "
                "WHERE commands_json LIKE '%Heal%' OR commands_json LIKE '%Restore%'"
            ):
                try:
                    cmds = json.loads(cj)
                    if any(c in {"Heal", "Restore", "Recover"} for c in cmds):
                        ids.add(cid.upper())
                except Exception:
                    pass
            return frozenset(ids)
        finally:
            conn.close()

    @lru_cache(maxsize=1)
    def aoe_cards(self) -> frozenset[str]:
        """所有 target_type=AllEnemies 的 attack cards。"""
        conn = self._connect()
        if conn is None:
            return frozenset()
        try:
            cur = conn.cursor()
            ids: set[str] = set()
            for (cid,) in cur.execute(
                "SELECT id FROM cards WHERE card_type='Attack' AND target_type LIKE '%All%'"
            ):
                ids.add(cid.upper())
            return frozenset(ids)
        finally:
            conn.close()

    def find_cards(
        self,
        card_type: str | None = None,      # Attack/Skill/Power
        rarity: str | None = None,          # common/uncommon/rare/basic/special
        target_type: str | None = None,     # SingleEnemy/AllEnemies/Self
        tag: str | None = None,             # 'aoe', 'block', 'damage', etc.
        limit: int = 20,
    ) -> list[str]:
        """按条件查找 card id 列表。纯数据查询，不硬编码卡名。"""
        conn = self._connect()
        if conn is None:
            return []
        try:
            cur = conn.cursor()
            where = []
            params: list = []
            if card_type:
                where.append("card_type=?"); params.append(card_type)
            if rarity:
                where.append("rarity=?"); params.append(rarity)
            if target_type:
                where.append("target_type LIKE ?"); params.append(f"%{target_type}%")
            if tag:
                where.append("tags_json LIKE ?"); params.append(f'%"{tag}"%')
            sql = "SELECT id FROM cards"
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += f" ORDER BY id LIMIT {limit}"
            return [r[0] for r in cur.execute(sql, params)]
        finally:
            conn.close()


# 全局单例（懒加载）
GAME_CATALOG = GameCatalog()


def attach_live_sim(client) -> None:
    """训练启动时调一次，让后续 encounter 查询走 sim API。"""
    GAME_CATALOG.attach_sim(client)
