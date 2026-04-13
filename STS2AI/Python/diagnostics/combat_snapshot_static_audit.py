#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


THIS_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = THIS_DIR.parent
REPO_ROOT = PYTHON_ROOT.parent.parent

FACADE_PATH = REPO_ROOT / "STS2AI" / "ENV" / "Sim" / "Overlay" / "Simulation" / "FullRunSimulatorRuntimeFacade.cs"
COMBAT_ROOM_PATH = REPO_ROOT / "STS2AI" / "ENV" / "Sim" / "Overlay" / "Compat" / "Upstream0991" / "Core" / "Rooms" / "CombatRoom.cs"
DTO_PATH = REPO_ROOT / "STS2AI" / "ENV" / "Sim" / "Overlay" / "Training" / "CombatTrainingDtos.cs"
DEFAULT_OUTPUT = REPO_ROOT / "STS2AI" / "Artifacts" / "tmp" / "combat_snapshot_static_audit.json"


@dataclass(frozen=True)
class CoverageItem:
    key: str
    label: str
    capture_patterns: tuple[str, ...]
    restore_patterns: tuple[str, ...]
    note: str = ""


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _find_block(text: str, anchor: str) -> str:
    start = text.find(anchor)
    if start < 0:
        return ""
    brace = text.find("{", start)
    if brace < 0:
        return ""
    depth = 0
    for index in range(brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def _extract_properties(block: str) -> list[str]:
    return re.findall(r"public\s+[^{;\n]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{\s*get;", block)


def _has_any(text: str, patterns: Iterable[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def _line_of(text: str, pattern: str) -> int | None:
    idx = text.find(pattern)
    if idx < 0:
        return None
    return text[:idx].count("\n") + 1


def build_report() -> dict[str, object]:
    facade_text = _read_text(FACADE_PATH)
    room_text = _read_text(COMBAT_ROOM_PATH)
    dto_text = _read_text(DTO_PATH)

    saved_snapshot_block = _find_block(facade_text, "public sealed class SavedCombatSnapshot")
    saved_player_block = _find_block(facade_text, "public sealed class SavedCombatPlayerSnapshot")
    capture_block = _find_block(facade_text, "private SavedCombatSnapshot? CaptureCombatSnapshot")
    apply_block = _find_block(facade_text, "private void ApplyCombatSnapshot")
    restore_pile_block = _find_block(facade_text, "private static void RestoreCombatPileState")
    restore_creature_block = _find_block(facade_text, "private static void RestoreCreatureState")
    restore_random_block = _find_block(facade_text, "private static void RestoreCombatRandomState")
    capture_random_block = _find_block(facade_text, "private static SavedCombatRandomStateSnapshot CaptureCombatRandomState")
    room_to_serializable = _find_block(room_text, "public override SerializableRoom ToSerializable()")
    room_from_serializable = _find_block(room_text, "public new static CombatRoom FromSerializable")
    combat_state_block = _find_block(dto_text, "public sealed class CombatTrainingStateSnapshot")
    combat_player_block = _find_block(dto_text, "public sealed class CombatTrainingPlayerSnapshot")
    combat_creature_block = _find_block(dto_text, "public sealed class CombatTrainingCreatureSnapshot")

    coverage_items = [
        CoverageItem("combat.round_number", "战斗轮数", ("RoundNumber =", "RoundNumber = combatRoom.CombatState.RoundNumber"), ("combatState.RoundNumber =",), ""),
        CoverageItem("combat.current_side", "当前回合方", ("CurrentSide =",), ("combatState.CurrentSide =",), ""),
        CoverageItem("combat.play_phase", "是否玩家出牌阶段", ("IsPlayPhase =",), ("CombatManagerIsPlayPhaseField.SetValue",), ""),
        CoverageItem("combat.player_actions_disabled", "玩家动作禁用标志", ("PlayerActionsDisabled =",), ("CombatManagerPlayerActionsDisabledField.SetValue",), ""),
        CoverageItem("combat.enemy_turn_started", "敌方回合已开始标志", ("IsEnemyTurnStarted =",), ("CombatManagerIsEnemyTurnStartedField.SetValue",), ""),
        CoverageItem("player.hp", "玩家当前血量", ("CurrentHp = player.Creature.CurrentHp",), ("SetCurrentHpInternal",), ""),
        CoverageItem("player.block", "玩家格挡", ("Block = player.Creature.Block",), ("Block = savedPlayer.CombatState.Block",), ""),
        CoverageItem("player.energy", "玩家能量", ("Energy = pcs.Energy",), ("pcs.Energy = saved.Energy",), ""),
        CoverageItem("player.hand_draw_discard_exhaust", "玩家手牌/抽牌/弃牌/消耗牌堆", ("Hand = pcs.Hand.Cards", "DrawPile = pcs.DrawPile.Cards", "DiscardPile = pcs.DiscardPile.Cards", "ExhaustPile = pcs.ExhaustPile.Cards"), ("RestoreCombatPileState(",), ""),
        CoverageItem("player.play_pile", "玩家出牌堆", ("PlayPile = pcs.PlayPile.Cards",), ("RestoreCombatPileState(",), ""),
        CoverageItem("player.powers", "玩家 powers", ("Powers = player.Creature.Powers",), ("RestoreCreatureState(player.Creature",), ""),
        CoverageItem("player.stars", "玩家星数", ("Stars = pcs.Stars",), ("pcs.Stars = savedPlayer.Stars",), ""),
        CoverageItem("player.potions", "玩家药水槽", ("Potions = player.PotionSlots",), ("RestorePlayerPotionState(",), "这轮刚补上"),
        CoverageItem("player.max_potion_count", "玩家药水槽上限", ("MaxPotionCount = player.MaxPotionCount",), ("RestorePlayerPotionState(",), "这轮刚补上"),
        CoverageItem("enemy.hp_block_powers", "敌人血量/格挡/powers", ("Hp = creature.CurrentHp", "Block = creature.Block", "Powers = creature.Powers"), ("RestoreCreatureState(creature, savedCreature)",), ""),
        CoverageItem("enemy.max_hp", "敌人最大血量", ("MaxHp = creature.MaxHp",), ("RestoreCreatureState(creature, savedCreature)",), ""),
        CoverageItem("enemy.identity", "敌人 combat_id / model_id", ("CombatId = creature.CombatId", "Id = creature.ModelId.Entry"), ("savedEnemyCombatIds", "savedEnemyIds"), ""),
        CoverageItem("enemy.move_state", "敌人下一步行动状态机", ("CurrentMoveId =", "StateLogIds =", "PerformedFirstMove =", "CurrentMovePerformedAtLeastOnce =", "SpawnedThisTurn ="), ("monster.SetMoveImmediate", "machine.StateLog.Clear()", "MonsterMoveStateMachinePerformedFirstMoveField.SetValue", "MonsterSpawnedThisTurnField.SetValue"), ""),
        CoverageItem("combat.random_state", "战斗随机数状态", ("RunRng =", "SharedRelicGrabBag =", "Players[player.NetId] = new SavedCombatPlayerRandomStateSnapshot"), ("RestoreCombatRandomState(runState", "player.PlayerRng.LoadFromSerializable", "player.PlayerOdds.LoadFromSerializable"), ""),
        CoverageItem("room.encounter_custom_state", "遭遇战自定义状态", ("EncounterState = Encounter.SaveCustomState()",), ("encounterModel.LoadCustomState(serializableRoom.EncounterState)",), "走 CombatRoomSnapshot"),
        CoverageItem("room.extra_rewards", "额外奖励缓存", ("serializableRoom.ExtraRewards",), ("combatRoom._extraRewards.Add",), "走 CombatRoomSnapshot"),
        CoverageItem("selection.pending_selection", "战斗外待恢复选择", ("CapturePendingSelection()",), ("RestorePendingSelectionFlowAsync",), "不属于 SavedCombatSnapshot，走 bridge"),
        CoverageItem("combat.hand_selection", "战斗手牌选择状态", (), (), "当前不在 combat search snapshot；有单独 bridge/restore 路线"),
        CoverageItem("combat.card_selection", "战斗选卡状态", (), (), "当前不在 combat search snapshot；有单独 bridge/restore 路线"),
        CoverageItem("combat.action_queue_running", "动作队列运行中标志", (), (), "未看到显式 capture/restore"),
        CoverageItem("combat.can_end_turn", "是否可结束回合", (), (), "未看到显式 capture/restore，可能由 manager 状态派生"),
        CoverageItem("player.relic_runtime_state", "遗物运行时战斗状态", (), (), "未看到显式 capture/restore，高风险漏项"),
        CoverageItem("player.extra_fields", "玩家 ExtraFields / 额外运行时字段", (), (), "未看到显式 capture/restore，高风险漏项"),
    ]

    item_reports: list[dict[str, object]] = []
    for item in coverage_items:
        capture_hit = _has_any(capture_block, item.capture_patterns) or _has_any(room_to_serializable, item.capture_patterns) or _has_any(capture_random_block, item.capture_patterns)
        restore_hit = _has_any(apply_block, item.restore_patterns) or _has_any(restore_pile_block, item.restore_patterns) or _has_any(restore_creature_block, item.restore_patterns) or _has_any(room_from_serializable, item.restore_patterns) or _has_any(restore_random_block, item.restore_patterns) or _has_any(facade_text, item.restore_patterns)
        if capture_hit and restore_hit:
            status = "covered"
        elif capture_hit or restore_hit:
            status = "partial"
        else:
            status = "missing_or_implicit"
        item_reports.append(
            {
                "key": item.key,
                "label": item.label,
                "status": status,
                "capture_hit": capture_hit,
                "restore_hit": restore_hit,
                "note": item.note,
            }
        )

    high_risk = [item for item in item_reports if item["status"] != "covered" and ("高风险" in str(item["note"]) or str(item["key"]).startswith(("player.relic", "player.extra_fields", "combat.action_queue", "combat.can_end_turn")))]

    return {
        "paths": {
            "facade": str(FACADE_PATH),
            "combat_room": str(COMBAT_ROOM_PATH),
            "combat_dto": str(DTO_PATH),
        },
        "snapshot_schema": {
            "saved_combat_snapshot_fields": _extract_properties(saved_snapshot_block),
            "saved_combat_player_fields": _extract_properties(saved_player_block),
            "combat_training_state_fields": _extract_properties(combat_state_block),
            "combat_training_player_fields": _extract_properties(combat_player_block),
            "combat_training_creature_fields": _extract_properties(combat_creature_block),
        },
        "anchors": {
            "saved_combat_snapshot_line": _line_of(facade_text, "public sealed class SavedCombatSnapshot"),
            "capture_combat_snapshot_line": _line_of(facade_text, "private SavedCombatSnapshot? CaptureCombatSnapshot"),
            "apply_combat_snapshot_line": _line_of(facade_text, "private void ApplyCombatSnapshot"),
            "combat_room_to_serializable_line": _line_of(room_text, "public override SerializableRoom ToSerializable()"),
            "combat_room_from_serializable_line": _line_of(room_text, "public new static CombatRoom FromSerializable"),
        },
        "coverage_items": item_reports,
        "summary": {
            "covered": sum(1 for item in item_reports if item["status"] == "covered"),
            "partial": sum(1 for item in item_reports if item["status"] == "partial"),
            "missing_or_implicit": sum(1 for item in item_reports if item["status"] == "missing_or_implicit"),
            "high_risk_gaps": high_risk,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="静态审计 combat search snapshot 覆盖面")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 JSON 路径")
    args = parser.parse_args()

    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
