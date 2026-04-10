using System.Collections.Generic;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Models;

namespace MegaCrit.Sts2.Core.TestSupport;

public interface IHandSelectionCardSelector : ICardSelectionPromptSelector
{
	void RegisterHandSelection(IEnumerable<CardModel> options, CardSelectorPrefs prefs, string mode);
}
