"""sim 与原游戏桥接的一致性报告。"""

from __future__ import annotations

from typing import Any

from game_bridge.session.state_semantics import is_actionable_combat_state, is_combat_state, state_type


def static_consistency_report() -> dict[str, Any]:
    checks = [
        {
            "area": "full_run/http_v2",
            "status": "supported",
            "details": "支持 state/reset/step/batch_step/save/load/import/export/delete_state。",
        },
        {
            "area": "full_run/pipe_proto",
            "status": "supported",
            "details": "支持 reset/state/step/batch_step/save/load/import/export/delete/perf/local_ort/mcts。",
        },
        {
            "area": "combat/pipe_proto",
            "status": "supported",
            "details": "支持 combat_reset/combat_state/combat_step，并以 sim legal_actions 为权威来源。",
        },
        {
            "area": "catalog/proto_pipe",
            "status": "missing",
            "details": "proto pipe 仍缺 game_catalog/combat_catalog opcode，当前依赖 sqlite 或 json RPC fallback。",
        },
        {
            "area": "full_run/http_local_ort",
            "status": "unsupported",
            "details": "HTTP 路径不支持本地 ORT rollout 与 combat MCTS，只在 proto pipe 暴露。",
        },
        {
            "area": "state_semantics",
            "status": "partial",
            "details": "已统一 combat/actionable/menu-ready 判定；但缺少对 live 游戏逐 screen 的自动对拍脚本。",
        },
    ]
    return {
        "summary": {
            "supported": sum(1 for item in checks if item["status"] == "supported"),
            "partial": sum(1 for item in checks if item["status"] == "partial"),
            "missing": sum(1 for item in checks if item["status"] in {"missing", "unsupported"}),
        },
        "checks": checks,
        "next_actions": [
            "为 proto pipe 增加 game_catalog/combat_catalog opcode，去掉 catalog 层对 sqlite 的主路径依赖。",
            "补 live sim vs 原游戏的逐 screen 回归脚本，覆盖 map/event/shop/rest/card_select/combat_rewards。",
            "把 reset/step 的 failure code 与 screen invariant 输出到统一诊断报告，而不是只在异常对象上挂属性。",
        ],
    }


def inspect_state_consistency(state: dict[str, Any]) -> dict[str, Any]:
    current_type = state_type(state)
    issues: list[str] = []
    if not current_type:
        issues.append("state_type 缺失。")
    if "legal_actions" not in state:
        issues.append("顶层 legal_actions 缺失。")
    if is_combat_state(state):
        if "battle" not in state:
            issues.append("combat state 缺 battle 字段。")
        if not isinstance((state.get("battle") or {}).get("enemies"), list):
            issues.append("battle.enemies 缺失或类型错误。")
        if not isinstance((state.get("battle") or {}).get("hand"), list):
            issues.append("battle.hand 缺失或类型错误。")
        if not is_actionable_combat_state(state) and not bool(state.get("terminal")):
            issues.append("当前 combat state 非可行动，调用方需要等待 settle 后再决策。")
    return {
        "state_type": current_type,
        "is_combat_state": is_combat_state(state),
        "issues": issues,
        "ok": not issues,
    }


def build_consistency_report(*, state: dict[str, Any] | None = None) -> dict[str, Any]:
    report = static_consistency_report()
    if state is not None:
        report["state_check"] = inspect_state_consistency(state)
    return report


__all__ = [
    "build_consistency_report",
    "inspect_state_consistency",
    "static_consistency_report",
]
