using System.Collections.Generic;

namespace MegaCrit.Sts2.Core.Simulation;

internal readonly record struct SelectionCardState(int Index, bool IsSelected);

internal static class SelectionActionSemantics
{
	public static bool IsQuotaReached(int selectedCount, int maxSelect)
	{
		return maxSelect > 0 && selectedCount >= maxSelect;
	}

	public static bool ShouldExposeSelectionActions(int selectedCount, int maxSelect, bool previewShowing = false)
	{
		return !previewShowing && !IsQuotaReached(selectedCount, maxSelect);
	}

	public static HashSet<int> CollectSelectedIndices(
		IEnumerable<int>? explicitSelectedIndices = null,
		IEnumerable<SelectionCardState>? cards = null)
	{
		HashSet<int> selectedIndices = new HashSet<int>();

		if (cards != null)
		{
			foreach (SelectionCardState card in cards)
			{
				if (card.IsSelected && card.Index >= 0)
				{
					selectedIndices.Add(card.Index);
				}
			}
		}

		if (explicitSelectedIndices != null)
		{
			foreach (int index in explicitSelectedIndices)
			{
				if (index >= 0)
				{
					selectedIndices.Add(index);
				}
			}
		}

		return selectedIndices;
	}
}
