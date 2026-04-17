"""Route Compiler: 地图路线选择。

游戏数据格式 (map state):
  {
    "next_options": [
      {"index": int, "col": int, "row": int, "point_type": str, "label": str},
      ...
    ],
    "nodes": [...],  // 完整地图拓扑
  }
"""

from __future__ import annotations

from typing import Any

from networkV2.s1_schema.actions import ActionCandidate


# 节点类型风险/价值标注（U8 修复后直接存原始 [0,1] 值）
# 历史：原先值 × 30/50 是为了对抗 _action_token 把 damage_est/block_est 做归一化
# （_DMG=30 / _BLK=50），这是"字段语义黑客"。现在 ActionCandidate 有 route_risk /
# route_value 专属字段，bank_assembler 直接透传原值不做归一化，常数 × 30/50 去掉。
_NODE_RISK = {
    "monster": 0.3, "elite": 0.7, "boss": 1.0,
    "event": 0.2,   "treasure": 0.0,
    "rest_site": 0.0, "shop": 0.0,
}
_NODE_VALUE = {
    "monster": 0.3, "elite": 0.6, "boss": 0.0,
    "event": 0.4,   "treasure": 0.5,
    "rest_site": 0.7, "shop": 0.6,
}


class RouteCompiler:
    """编译 route 选项为 ActionCandidate 列表。"""

    def compile(
        self,
        obs: dict[str, Any],
        legal_actions: list[dict[str, Any]],
    ) -> list[ActionCandidate]:
        candidates: list[ActionCandidate] = []
        map_state = obs.get("map") or {}
        options_by_index = {
            opt.get("index", i): opt
            for i, opt in enumerate(map_state.get("next_options", []) or [])
            if isinstance(opt, dict)
        }

        for i, action in enumerate(legal_actions):
            if not isinstance(action, dict):
                continue
            action_type = str(action.get("action", "") or "").lower()

            if action_type == "choose_map_node":
                idx = action.get("index", -1)
                opt = options_by_index.get(idx, {})
                point_type = str(opt.get("point_type", opt.get("type", "")) or "").lower()
                label = str(action.get("label", "") or "") or point_type

                roles = [point_type] if point_type else []

                candidates.append(ActionCandidate(
                    action_type=action_type,
                    action_index=i,
                    label=label,
                    family="map",
                    target_scope="map",
                    roles=roles,
                    # U8: 直接写 route 专属字段（[0,1]），不再复用 damage_est/block_est
                    route_risk=_NODE_RISK.get(point_type, 0.3),
                    route_value=_NODE_VALUE.get(point_type, 0.3),
                ))

        return candidates
