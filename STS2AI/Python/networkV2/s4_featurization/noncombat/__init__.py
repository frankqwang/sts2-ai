"""非战斗 Option Builder：card_reward / shop / route / rest / event。

每个 domain builder 负责：
  1. 从 obs 中提取该 domain 的选项候选
  2. 构建为 ActionCandidate (复用 action_bank token 类型)
  3. 返回选项列表，由 decision_featurizer 放入 action_bank

Shared banks (build/inventory/economy/route/objective/forecast) 由 token_bank_builder 统一处理。
"""

from networkV2.s4_featurization.noncombat.card_reward_options import CardRewardOptionBuilder
from networkV2.s4_featurization.noncombat.shop_options import ShopOptionBuilder
from networkV2.s4_featurization.noncombat.route_options import RouteOptionBuilder
from networkV2.s4_featurization.noncombat.rest_options import RestOptionBuilder
from networkV2.s4_featurization.noncombat.event_options import EventOptionBuilder
from networkV2.s4_featurization.noncombat.selection_options import SelectionOptionBuilder
