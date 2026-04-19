"""Runtime 提取器：从 bridge obs 提取运行时状态。

对齐 proto state dict 和 combat_training_env 的 normalize 输出。
历史上也支持 binary_pipe_client snapshot,2026-04-18 binary wire 已废弃。
"""

from __future__ import annotations

from typing import Any

from networkV2.s1_schema.entities import (
    CardSemantics,
    RelicSemantics,
    PotionSemantics,
    PlayerRuntime,
    HandCardRuntime,
    EnemyRuntime,
    IntentInfo,
    PileSummary,
)
from networkV2.s1_schema.card_semantic_catalog import CARD_SEMANTICS
from networkV2.s4_featurization.card_base_stats import base_damage, base_hits, base_block, is_aoe


def _pick(raw: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    if not isinstance(raw, dict):
        return default
    for key in keys:
        if key in raw:
            return raw[key]
    return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val or default)
    except (TypeError, ValueError):
        return default


class RuntimeExtractor:
    """从 bridge obs 编译 RuntimeInstances。

    兼容两种数据源:
      1. proto GameState → dict (top-level 有 battle dict,和老 binary_pipe_client
         snapshot 同 shape,V2 训练主路径)
      2. combat_training_env normalize 后的 flat snapshot (hand/enemies 在 top-level)
    """

    def extract(self, obs: dict[str, Any]) -> tuple[
        PlayerRuntime,
        list[HandCardRuntime],
        list[EnemyRuntime],
        list[PileSummary],
        list[CardSemantics],
        list[RelicSemantics],
        list[PotionSemantics],
        dict[str, Any],  # combat_meta
    ]:
        """编译全部运行时实例。

        Returns:
            player, hand_cards, enemies, piles, deck_cards, relics, potions, combat_meta
        """
        # 优先从 battle dict 取战斗数据，fallback 到 top-level
        battle = obs.get("battle") or {}

        player = self._compile_player(obs, battle)
        hand_cards = self._compile_hand(obs, battle)
        enemies = self._compile_enemies(obs, battle)
        piles = self._compile_piles(obs, battle)
        deck_cards = self._compile_deck(obs)
        relics = self._compile_relics(obs)
        potions = self._compile_potions(obs)
        combat_meta = self._compile_combat_meta(obs, battle)

        return player, hand_cards, enemies, piles, deck_cards, relics, potions, combat_meta

    # ------------------------------------------------------------------
    # Player
    # ------------------------------------------------------------------

    def _compile_player(self, obs: dict[str, Any], battle: dict[str, Any]) -> PlayerRuntime:
        # 优先从 battle.player 取（含 powers），fallback 到 top-level player
        raw = battle.get("player") or obs.get("player") or {}

        powers_dict = self._parse_powers(raw)

        return PlayerRuntime(
            hp=_safe_int(_pick(raw, "hp", "current_hp")),
            max_hp=max(_safe_int(_pick(raw, "max_hp", default=1)), 1),
            block=_safe_int(_pick(raw, "block")),
            energy=_safe_int(_pick(raw, "energy") or _pick(battle, "energy")),
            max_energy=_safe_int(_pick(raw, "max_energy") or _pick(battle, "max_energy", default=3)),
            powers=powers_dict,
        )

    # ------------------------------------------------------------------
    # Hand
    # ------------------------------------------------------------------

    def _compile_hand(self, obs: dict[str, Any], battle: dict[str, Any]) -> list[HandCardRuntime]:
        raw_hand = obs.get("hand") or battle.get("hand") or []
        if not isinstance(raw_hand, list):
            raw_hand = []

        cards: list[HandCardRuntime] = []
        for i, raw_card in enumerate(raw_hand):
            if not isinstance(raw_card, dict):
                continue
            card_id = str(_pick(raw_card, "id", default="") or "").lower()
            snap = CARD_SEMANTICS.get(card_id)
            # card_type: 游戏发 "ATTACK"/"SKILL"/"POWER" 或 int 0-6
            card_type = self._normalize_card_type(
                _pick(raw_card, "type", "card_type", "CardType", default="")
            ) or snap.card_type
            runtime_keywords = [
                str(k).lower()
                for k in (_pick(raw_card, "keywords", default=[]) or [])
            ]
            keywords = sorted(set(runtime_keywords) | set(snap.db_keywords))
            # preview 数值：优先读 obs 透传字段，否则用 card_base_stats 估计
            dmg = _pick(raw_card, "damage", "damage_est", "damage_preview")
            blk = _pick(raw_card, "block", "block_est", "block_preview")
            drw = _pick(raw_card, "draw", "draw_est", "cards_drawn")
            damage_est = float(dmg) if dmg is not None else float(
                base_damage(card_id) * max(base_hits(card_id), 1)
            )
            block_est = float(blk) if blk is not None else float(base_block(card_id))
            # 若 obs 未给 draw，根据关键字做保守估计（"draw" 关键字默认 +1）
            if drw is not None:
                draw_est = float(drw)
            elif "draw" in keywords or snap.has_functional_tag("draw"):
                draw_est = 1.0
            else:
                draw_est = 0.0

            # upgrade_count：有些 bridge 直接给整数，有些给 bool
            upg_raw = _pick(raw_card, "upgrades", "upgrade_count", "UpgradeCount", default=None)
            if upg_raw is None:
                upg_count = 1 if bool(_pick(raw_card, "is_upgraded", default=False)) else 0
            else:
                try:
                    upg_count = int(upg_raw)
                except (TypeError, ValueError):
                    upg_count = 1 if bool(upg_raw) else 0
            # target_type：normalize 为小写字符串
            tt_raw = _pick(raw_card, "target_type", "TargetType", default="")
            target_type = (
                str(tt_raw or "").lower() if not isinstance(tt_raw, int) else f"type_{tt_raw}"
            ) or snap.target_type
            # rarity
            rarity = str(_pick(raw_card, "rarity", "Rarity", default="") or "").lower() or snap.rarity
            raw_cost = _pick(raw_card, "cost", "energy_cost", default=None)
            current_cost = _safe_int(raw_cost) if raw_cost is not None else int(snap.base_cost or 0)

            cards.append(HandCardRuntime(
                card_id=card_id,
                hand_index=_safe_int(_pick(raw_card, "index", "hand_index", default=i)),
                current_cost=current_cost,
                card_type=card_type,
                is_upgraded=upg_count > 0,
                upgrade_count=upg_count,
                rarity=rarity,
                target_type=target_type,
                can_play=bool(_pick(raw_card, "can_play", default=False)),
                requires_target=bool(_pick(raw_card, "requires_target", default=False)),
                valid_target_ids=[
                    int(t)
                    for t in (_pick(raw_card, "valid_target_ids", default=[]) or [])
                ],
                keywords=keywords,
                damage_est=damage_est,
                block_est=block_est,
                draw_est=draw_est,
                retain=("retain" in keywords) or snap.has_functional_tag("retain"),
                ethereal=("ethereal" in keywords) or snap.has_functional_tag("ethereal"),
                exhaust=("exhaust" in keywords) or snap.has_functional_tag("exhaust"),
            ))
        return cards

    # ------------------------------------------------------------------
    # Enemies
    # ------------------------------------------------------------------

    def _compile_enemies(self, obs: dict[str, Any], battle: dict[str, Any]) -> list[EnemyRuntime]:
        raw_enemies = obs.get("enemies") or battle.get("enemies") or obs.get("monsters") or []
        if not isinstance(raw_enemies, list):
            raw_enemies = []

        enemies: list[EnemyRuntime] = []
        for raw_enemy in raw_enemies:
            if not isinstance(raw_enemy, dict):
                continue
            if not bool(_pick(raw_enemy, "is_alive", default=True)):
                continue

            intents = self._parse_intents(raw_enemy)
            powers_dict = self._parse_powers(raw_enemy)

            enemy_id = str(_pick(raw_enemy, "id", "entity_id", "monster_id", default="") or "").lower()
            combat_id = _safe_int(_pick(raw_enemy, "combat_id", "target_id", default=-1))

            enemies.append(EnemyRuntime(
                entity_id=f"{enemy_id}_{combat_id}" if combat_id >= 0 else enemy_id,
                enemy_id=enemy_id,
                combat_id=combat_id,
                name=str(_pick(raw_enemy, "name", default=enemy_id) or enemy_id),
                hp=_safe_int(_pick(raw_enemy, "hp", "current_hp")),
                max_hp=max(_safe_int(_pick(raw_enemy, "max_hp", default=1)), 1),
                block=_safe_int(_pick(raw_enemy, "block")),
                is_alive=True,
                is_hittable=bool(_pick(raw_enemy, "is_hittable", default=True)),
                intends_to_attack=bool(_pick(raw_enemy, "intends_to_attack", default=False)),
                next_move_id=_pick(raw_enemy, "next_move_id"),
                intents=intents,
                powers=powers_dict,
            ))
        return enemies

    def _parse_intents(self, raw_enemy: dict[str, Any]) -> list[IntentInfo]:
        raw_intents = _pick(raw_enemy, "intents", default=[]) or []
        intents: list[IntentInfo] = []
        for raw in raw_intents:
            if not isinstance(raw, dict):
                continue
            intent_type = str(
                _pick(raw, "type", "intent_type", default="") or ""
            ).lower()
            damage = _safe_int(_pick(raw, "damage"))
            # binary_pipe 用 hits，normalize 后用 repeats
            hits = _safe_int(_pick(raw, "hits", "repeats", default=1))
            hits = max(hits, 1)
            total = _safe_int(_pick(raw, "total_damage"))
            if total == 0 and damage > 0:
                total = damage * hits

            intents.append(IntentInfo(
                intent_type=intent_type,
                damage=damage,
                total_damage=total,
                repeats=hits,
            ))

        # fallback: 旧格式可能用 top-level intent_type/intent_damage
        if not intents and "intent_type" in raw_enemy:
            intents.append(IntentInfo(
                intent_type=str(raw_enemy.get("intent_type", "") or "").lower(),
                damage=_safe_int(raw_enemy.get("intent_damage")),
                total_damage=_safe_int(raw_enemy.get("intent_damage", 0)) * max(
                    _safe_int(raw_enemy.get("intent_hits", 1)), 1
                ),
                repeats=max(_safe_int(raw_enemy.get("intent_hits", 1)), 1),
            ))
        return intents

    def _parse_powers(self, raw: dict[str, Any]) -> dict[str, int]:
        """解析 powers 列表。兼容 powers/status/buffs 字段名。

        归一化 power ID：
          STRENGTH_POWER → strength
          VULNERABLE_POWER → vulnerable
          SomePower → somepower (直接 lower)
        """
        raw_list = _pick(raw, "powers", "status", "buffs", default=[]) or []
        result: dict[str, int] = {}
        for p in raw_list:
            if not isinstance(p, dict):
                continue
            pid = str(_pick(p, "id", default="") or "")
            amt = _safe_int(_pick(p, "amount", default=0))
            if not pid or amt == 0:
                continue
            # 归一化: 去掉 _POWER/_BUFF 后缀，转小写
            normalized = self._normalize_power_id(pid)
            result[normalized] = amt
        return result

    @staticmethod
    def _normalize_power_id(raw_id: str) -> str:
        """STRENGTH_POWER → strength, HardenedShell → hardenedshell"""
        s = raw_id.strip()
        # 去掉常见后缀
        for suffix in ("_POWER", "_BUFF", "_DEBUFF", "Power", "Buff", "Debuff"):
            if s.endswith(suffix):
                s = s[:-len(suffix)]
                break
        return s.lower().strip("_")

    # ------------------------------------------------------------------
    # Piles
    # ------------------------------------------------------------------

    def _compile_piles(self, obs: dict[str, Any], battle: dict[str, Any]) -> list[PileSummary]:
        piles: list[PileSummary] = []

        # 游戏发送的真实 key: draw_pile_cards / discard_pile_cards / exhaust_pile_cards
        pile_configs = [
            ("draw",    ["draw_pile_cards"],    ["draw_pile_count", "draw_pile_size"]),
            ("discard", ["discard_pile_cards"], ["discard_pile_count", "discard_pile_size"]),
            ("exhaust", ["exhaust_pile_cards"], ["exhaust_pile_count", "exhaust_pile_size"]),
        ]

        for pile_type, card_keys, count_keys in pile_configs:
            # 优先从 battle dict 拿卡牌列表
            card_ids: list[str] | None = None
            for source in (battle, obs):
                for k in card_keys:
                    val = source.get(k)
                    if isinstance(val, list):
                        card_ids = val
                        break
                if card_ids is not None:
                    break

            if card_ids is not None:
                # 卡牌列表可能是 str list 或 dict list
                piles.append(self._pile_from_card_ids(pile_type, card_ids))
            else:
                # fallback: 只有 count
                size = 0
                for source in (battle, obs):
                    for k in count_keys:
                        if k in source:
                            size = _safe_int(source[k])
                            break
                    if size > 0:
                        break
                # player 里也可能有 count
                player = obs.get("player") or {}
                if size == 0:
                    for k in count_keys:
                        if k in player:
                            size = _safe_int(player[k])
                            break
                piles.append(PileSummary(pile_type=pile_type, size=size))

        return piles

    def _pile_from_card_ids(self, pile_type: str, cards: list) -> PileSummary:
        """从卡牌 ID 列表构建 PileSummary。"""
        attack_count = 0
        skill_count = 0
        power_count = 0
        curse_count = 0
        status_count = 0
        zero_cost_count = 0
        card_ids: list[str] = []

        for c in cards:
            card_id = ""
            card_type = ""
            cost: int | None = None
            if isinstance(c, str):
                card_id = c.lower()
            elif isinstance(c, dict):
                card_id = str(_pick(c, "id", default="") or "").lower()
                card_type = self._normalize_card_type(_pick(c, "type", "card_type", default=""))
                cost = _safe_int(_pick(c, "cost", "energy_cost"))
            if not card_id:
                continue
            snap = CARD_SEMANTICS.get(card_id)
            if not card_type:
                card_type = snap.card_type
            if cost is None:
                cost = snap.base_cost
            card_ids.append(card_id)
            if card_type == "attack":
                attack_count += 1
            elif card_type == "skill":
                skill_count += 1
            elif card_type == "power":
                power_count += 1
            elif card_type == "curse":
                curse_count += 1
            elif card_type == "status":
                status_count += 1
            if int(cost or 0) == 0:
                zero_cost_count += 1

        return PileSummary(
            pile_type=pile_type,
            size=len(cards),
            attack_count=attack_count,
            skill_count=skill_count,
            power_count=power_count,
            curse_count=curse_count,
            status_count=status_count,
            zero_cost_count=zero_cost_count,
            card_ids=card_ids,
        )

    # ------------------------------------------------------------------
    # Deck / Relics / Potions (从 player static 编译)
    # ------------------------------------------------------------------

    def _compile_deck(self, obs: dict[str, Any]) -> list[CardSemantics]:
        player = obs.get("player") or {}
        raw_deck = player.get("deck") or []
        if not isinstance(raw_deck, list):
            return []

        cards: list[CardSemantics] = []
        for raw_card in raw_deck:
            if not isinstance(raw_card, dict):
                continue
            card_id = str(_pick(raw_card, "id", default="") or "").lower()
            snap = CARD_SEMANTICS.get(card_id)
            card_type = self._normalize_card_type(_pick(raw_card, "type", "card_type", default=""))
            rarity = str(_pick(raw_card, "rarity", default="") or "").lower()
            raw_cost = _pick(raw_card, "cost", "energy_cost", default=None)
            base_cost = _safe_int(raw_cost) if raw_cost is not None else snap.base_cost
            cards.append(CardSemantics(
                entity_id=card_id,
                card_type=card_type or snap.card_type,
                rarity=rarity or snap.rarity,
                base_cost=base_cost,
                is_upgraded=bool(_pick(raw_card, "is_upgraded", default=False)),
                tags=list(snap.db_tags),
                keywords=list(snap.db_keywords),
            ))
        return cards

    def _compile_relics(self, obs: dict[str, Any]) -> list[RelicSemantics]:
        player = obs.get("player") or {}
        raw_relics = player.get("relics") or []
        if not isinstance(raw_relics, list):
            return []

        return [
            RelicSemantics(entity_id=str(_pick(r, "id", default="") or "").lower())
            for r in raw_relics if isinstance(r, dict)
        ]

    def _compile_potions(self, obs: dict[str, Any]) -> list[PotionSemantics]:
        player = obs.get("player") or {}
        raw_potions = player.get("potions") or []
        if not isinstance(raw_potions, list):
            return []

        return [
            PotionSemantics(entity_id=str(_pick(p, "id", default="") or "").lower())
            for p in raw_potions if isinstance(p, dict)
        ]

    # ------------------------------------------------------------------
    # Combat metadata
    # ------------------------------------------------------------------

    def _compile_combat_meta(self, obs: dict[str, Any], battle: dict[str, Any]) -> dict[str, Any]:
        """提取战斗元信息。"""
        return {
            "round_number": _safe_int(
                _pick(battle, "round_number_raw") or _pick(obs, "round_number_raw", "turn", "round")
            ),
            "can_end_turn": bool(
                _pick(battle, "can_end_turn") or _pick(obs, "can_end_turn", default=False)
            ),
            "encounter_id": str(
                _pick(obs, "encounter_id", default="") or ""
            ).lower(),
            "state_type": str(
                _pick(obs, "state_type", default="") or ""
            ).lower(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _CARD_TYPE_MAP = {
        "0": "unknown", "1": "attack", "2": "skill", "3": "power",
        "4": "status", "5": "curse", "6": "quest",
        "attack": "attack", "skill": "skill", "power": "power",
        "status": "status", "curse": "curse", "quest": "quest",
        "unknown": "unknown",
    }

    def _normalize_card_type(self, raw: Any) -> str:
        return self._CARD_TYPE_MAP.get(str(raw).lower().strip(), "unknown")
