using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Entities.UI;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Unlocks;

namespace MegaCrit.Sts2.Core.Nodes.Screens.PotionLab;

public partial class NPotionLabCategory : VBoxContainer
{
	private MegaRichTextLabel _headerLabel;

	private GridContainer _potionContainer;

	public Control? DefaultFocusedControl
	{
		get
		{
			if (_potionContainer.GetChildCount() <= 0)
			{
				return null;
			}
			return _potionContainer.GetChild<Control>(0);
		}
	}

	public override void _Ready()
	{
		_headerLabel = GetNode<MegaRichTextLabel>("Header");
		_potionContainer = GetNode<GridContainer>("%PotionsContainer");
	}

	public void LoadPotions(PotionRarity potionRarity, LocString header, HashSet<PotionModel> seenPotions, UnlockState unlockState, HashSet<PotionModel> allUnlockedPotions, PotionRarity? secondRarity = null)
	{
		_headerLabel.Text = header.GetFormattedText();
		IEnumerable<PotionModel> enumerable = ModelDb.AllPotions.Where((PotionModel relic) => relic.Rarity == potionRarity || relic.Rarity == secondRarity);
		List<PotionModel> list = new List<PotionModel>();
		List<PotionModel> list2 = new List<PotionModel>();
		foreach (PotionPoolModel allCharacterPotionPool in ModelDb.AllCharacterPotionPools)
		{
			foreach (PotionModel item in enumerable)
			{
				if (allCharacterPotionPool.AllPotionIds.Contains(item.Id))
				{
					list.Add(item);
				}
			}
		}
		foreach (PotionModel item2 in enumerable)
		{
			if (!list.Contains(item2))
			{
				list2.Add(item2);
			}
		}
		StringComparer comparer = StringComparer.Create(LocManager.Instance.CultureInfo, CompareOptions.None);
		list2.Sort((PotionModel p1, PotionModel p2) => comparer.Compare(p1.Title.GetFormattedText(), p2.Title.GetFormattedText()));
		foreach (PotionModel item3 in list2.Concat(list))
		{
			ModelVisibility visibility = ((!allUnlockedPotions.Contains(item3)) ? ModelVisibility.Locked : (seenPotions.Contains(item3) ? ModelVisibility.Visible : ModelVisibility.NotSeen));
			NLabPotionHolder child = NLabPotionHolder.Create(item3.ToMutable(), visibility);
			_potionContainer.AddChildSafely(child);
		}
	}

	public void ClearPotions()
	{
		foreach (Node child in _potionContainer.GetChildren())
		{
			child.QueueFreeSafely();
		}
	}

	public List<IReadOnlyList<Control>> GetGridItems()
	{
		List<IReadOnlyList<Control>> list = new List<IReadOnlyList<Control>>();
		for (int i = 0; i < _potionContainer.GetChildren().Count; i += _potionContainer.Columns)
		{
			list.Add(_potionContainer.GetChildren().OfType<Control>().Skip(i)
				.Take(_potionContainer.Columns)
				.ToList());
		}
		return list;
	}
}
