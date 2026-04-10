using System.Collections.Generic;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Models;

namespace MegaCrit.Sts2.Core.TestSupport;

public interface IRelicSelectionPromptSelector : ICardSelector
{
	Task<RelicModel?> GetSelectedRelicAsync(IReadOnlyList<RelicModel> relics, bool canSkip);
}
