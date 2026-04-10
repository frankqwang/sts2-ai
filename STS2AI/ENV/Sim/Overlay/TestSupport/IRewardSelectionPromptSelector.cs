using System.Collections.Generic;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Rewards;

namespace MegaCrit.Sts2.Core.TestSupport;

public interface IRewardSelectionPromptSelector : ICardSelector
{
	Task<Reward?> GetSelectedRewardAsync(IReadOnlyList<Reward> rewards, bool canProceed);
}
