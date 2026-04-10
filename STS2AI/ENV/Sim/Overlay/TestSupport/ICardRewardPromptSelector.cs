using System.Collections.Generic;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Models;

namespace MegaCrit.Sts2.Core.TestSupport;

public interface ICardRewardPromptSelector : ICardSelector
{
	Task<CardModel?> GetSelectedCardRewardAsync(
		IReadOnlyList<CardCreationResult> options,
		IReadOnlyList<CardRewardAlternative> alternatives,
		bool canSkip);
}
