using System.Collections.Generic;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Models;

namespace MegaCrit.Sts2.Core.TestSupport;

public interface ICardSelectionPromptSelector : ICardSelector
{
	void RegisterCardSelection(IEnumerable<CardModel> options, CardSelectorPrefs prefs, string mode);
}
