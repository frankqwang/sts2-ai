"""非战斗 Compiler：card_reward / shop / route / rest / event。

每个 domain compiler 负责：
  1. 从 obs 中提取该 domain 的选项候选
  2. 编译为 ActionCandidate (复用 action_bank token 类型)
  3. 返回选项列表，由 feature_compiler 放入 action_bank

Shared banks (build/inventory/economy/route/objective/forecast) 由 bank_assembler 统一处理。
"""

from networkV2.s4_compiler.noncombat.card_reward_compiler import CardRewardCompiler
from networkV2.s4_compiler.noncombat.shop_compiler import ShopCompiler
from networkV2.s4_compiler.noncombat.route_compiler import RouteCompiler
from networkV2.s4_compiler.noncombat.rest_compiler import RestCompiler
from networkV2.s4_compiler.noncombat.event_compiler import EventCompiler
