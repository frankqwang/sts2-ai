"""Multi-domain Policy Head:每种决策 domain 独立 scoring,共享 decision_repr。

为什么拆:
  - 不同 domain 的 option 语义完全不同:
      * card_reward → 比较"这张卡在当前 build 里的边际价值"
      * map         → 比较"这个节点未来 ROI"
      * relic       → 比较"遗物 effect 合拍度"
      * campfire    → 比较"rest vs upgrade vs other"
  - 单个 PolicyHead 共享权重会让几种语义互相污染
  - 独立 head 成本低(PolicyHead 就一对 Bilinear+Linear),效果好

Domain 列表和 networkV2.s1_schema.token_banks.DECISION_DOMAINS 对齐:
  combat / card_reward / shop / route / rest / event / selection

向后兼容:
  `__call__(decision_repr, action_bt, domain=None)` 不传 domain 时走 "combat" head。
  在 UnifiedNet 里 forward 分支自然把 domain 传进来。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from networkV2.s5_net.heads.policy_head import PolicyHead
from networkV2.s5_net.tokenizer import BankTensor


# 实际 networkV2 使用的 7 个 domain
POLICY_DOMAINS: tuple[str, ...] = (
    "combat",
    "card_reward",
    "shop",
    "route",
    "rest",
    "event",
    "selection",
)


class MultiDomainPolicyHead(nn.Module):
    """per-domain 独立 scoring。"""

    def __init__(self, d_model: int = 384, domains: tuple[str, ...] = POLICY_DOMAINS) -> None:
        super().__init__()
        self.domains = domains
        self.heads = nn.ModuleDict({d: PolicyHead(d_model) for d in domains})

    def forward(
        self,
        decision_repr: torch.Tensor,
        action_refined: BankTensor,
        domain: str = "combat",
    ) -> torch.Tensor:
        # nn.ModuleDict 没有 dict.get;手动 fallback 到 combat
        if domain in self.heads:
            head = self.heads[domain]
        else:
            head = self.heads["combat"]
        return head(decision_repr, action_refined)

    def head(self, domain: str) -> PolicyHead:
        if domain in self.heads:
            return self.heads[domain]
        return self.heads["combat"]

    def load_from_shared_head(self, shared: PolicyHead) -> None:
        """把原 shared PolicyHead 的参数复制到所有 domain head(warm start)。

        用途:老 checkpoint 只有 `policy_head.*`,没有 `multi_domain_policy_head.heads.<d>.*`。
        调用方在 load_compatible_params 后调这个,让所有 domain 从 shared baseline 起步。
        """
        sd = shared.state_dict()
        for d, h in self.heads.items():
            h.load_state_dict(sd)
