using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Relics;
using MegaCrit.Sts2.Core.Entities.UI;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Acts;
using MegaCrit.Sts2.Core.Models.Relics;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Unlocks;

namespace MegaCrit.Sts2.Core.Nodes.Screens.RelicCollection;

public partial class NRelicCollectionCategory : VBoxContainer
{
	public static readonly string scenePath = SceneHelper.GetScenePath("screens/relic_collection/relic_collection_subcategory");

	private static readonly List<RelicModel> _relicModelCache = new List<RelicModel>();

	private NRelicCollection _collection;

	private MegaRichTextLabel _headerLabel;

	private GridContainer _relicsContainer;

	private readonly List<NRelicCollectionCategory> _subCategories = new List<NRelicCollectionCategory>();

	private Control _spacer;

	private TextureRect _icon;

	public Control? DefaultFocusedControl
	{
		get
		{
			if (_subCategories.Any())
			{
				return _subCategories.First().DefaultFocusedControl;
			}
			if (GodotObject.IsInstanceValid(_relicsContainer) && _relicsContainer.GetChildCount() > 0)
			{
				return _relicsContainer.GetChild<Control>(0);
			}
			return null;
		}
	}

	private NRelicCollectionCategory CreateForSubcategory()
	{
		return PreloadManager.Cache.GetScene(scenePath).Instantiate<NRelicCollectionCategory>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		_headerLabel = GetNode<MegaRichTextLabel>("%Header");
		_icon = GetNode<TextureRect>("%Icon");
		_relicsContainer = GetNode<GridContainer>("%RelicsContainer");
		_spacer = GetNode<Control>("Spacer");
		_icon.Visible = false;
	}

	public void LoadRelics(RelicRarity relicRarity, NRelicCollection collection, LocString header, HashSet<RelicModel> seenRelics, UnlockState unlockState, HashSet<RelicModel> allUnlockedRelics)
	{
		_subCategories.Clear();
		_headerLabel.Text = header.GetFormattedText();
		_collection = collection;
		_relicModelCache.Clear();
		_relicModelCache.AddRange(ModelDb.AllRelics.Where((RelicModel relic) => relic.Rarity == relicRarity));
		if (relicRarity == RelicRarity.Starter)
		{
			List<RelicModel> list = ModelDb.AllCharacters.SelectMany((CharacterModel c) => c.StartingRelics).ToList();
			NRelicCollectionCategory nRelicCollectionCategory = CreateForSubcategory();
			_subCategories.Add(nRelicCollectionCategory);
			this.AddChildSafely(nRelicCollectionCategory);
			MoveChild(nRelicCollectionCategory, _headerLabel.GetIndex() + 1);
			nRelicCollectionCategory._spacer.Visible = false;
			nRelicCollectionCategory._headerLabel.Visible = false;
			nRelicCollectionCategory.LoadSubcategory(_collection, null, list, seenRelics, allUnlockedRelics);
			IEnumerable<RelicModel> relics = list.Select((RelicModel r) => ModelDb.Relic<TouchOfOrobas>().GetUpgradedStarterRelic(r));
			NRelicCollectionCategory nRelicCollectionCategory2 = CreateForSubcategory();
			_subCategories.Add(nRelicCollectionCategory2);
			this.AddChildSafely(nRelicCollectionCategory2);
			MoveChild(nRelicCollectionCategory2, _headerLabel.GetIndex() + 2);
			nRelicCollectionCategory2._headerLabel.Visible = false;
			nRelicCollectionCategory2.LoadSubcategory(_collection, null, relics, seenRelics, allUnlockedRelics);
			return;
		}
		if (relicRarity == RelicRarity.Ancient)
		{
			int num = 4;
			List<ActModel> list2 = new List<ActModel>(num);
			CollectionsMarshal.SetCount(list2, num);
			Span<ActModel> span = CollectionsMarshal.AsSpan(list2);
			int num2 = 0;
			span[num2] = ModelDb.Act<Overgrowth>();
			num2++;
			span[num2] = ModelDb.Act<Underdocks>();
			num2++;
			span[num2] = ModelDb.Act<Hive>();
			num2++;
			span[num2] = ModelDb.Act<Glory>();
			List<ActModel> list3 = list2;
			if (ModelDb.Acts.Except(list3).Any())
			{
				throw new InvalidOperationException("The act list in NRelicCollectionCategory is out of date!");
			}
			List<AncientEventModel> list4 = list3.Select((ActModel a) => a.AllAncients).SelectMany((IEnumerable<AncientEventModel> a) => a).Concat(ModelDb.AllSharedAncients)
				.Distinct()
				.ToList();
			HashSet<AncientEventModel> hashSet = list3.Select((ActModel a) => a.GetUnlockedAncients(unlockState)).SelectMany((IEnumerable<AncientEventModel> a) => a).Concat(unlockState.SharedAncients)
				.Distinct()
				.ToHashSet();
			IReadOnlyDictionary<ModelId, AncientStats> ancientStats = SaveManager.Instance.Progress.AncientStats;
			LocString locString = new LocString("relic_collection", "UNKNOWN_ANCIENT");
			for (int num3 = 0; num3 < list4.Count; num3++)
			{
				AncientEventModel ancientEventModel = list4[num3];
				if (hashSet.Contains(ancientEventModel))
				{
					NRelicCollectionCategory nRelicCollectionCategory3 = CreateForSubcategory();
					_subCategories.Add(nRelicCollectionCategory3);
					this.AddChildSafely(nRelicCollectionCategory3);
					MoveChild(nRelicCollectionCategory3, _headerLabel.GetIndex() + num3 + 1);
					StringComparer comparer = StringComparer.Create(LocManager.Instance.CultureInfo, CompareOptions.None);
					RelicModel[] array = ancientEventModel.AllPossibleOptions.Select((EventOption o) => o.Relic?.CanonicalInstance).OfType<RelicModel>().Intersect(_relicModelCache)
						.OrderBy<RelicModel, string>((RelicModel r) => r.Title.GetFormattedText(), comparer)
						.ToArray();
					bool flag = ancientStats.ContainsKey(ancientEventModel.Id) || array.Any((RelicModel r) => seenRelics.Contains(r));
					LocString locString2 = new LocString("relic_collection", "ANCIENT_SUBCATEGORY");
					locString2.Add("Ancient", flag ? ancientEventModel.Title : locString);
					nRelicCollectionCategory3.LoadSubcategory(_collection, locString2, array, seenRelics, allUnlockedRelics);
					nRelicCollectionCategory3.LoadIcon(ancientEventModel.RunHistoryIcon);
				}
			}
			return;
		}
		List<RelicModel> list5 = new List<RelicModel>();
		List<RelicModel> list6 = new List<RelicModel>();
		foreach (RelicPoolModel allCharacterRelicPool in ModelDb.AllCharacterRelicPools)
		{
			foreach (RelicModel item in _relicModelCache)
			{
				if (allCharacterRelicPool.AllRelicIds.Contains(item.Id))
				{
					list5.Add(item);
				}
			}
		}
		foreach (RelicModel item2 in _relicModelCache)
		{
			if (!list5.Contains(item2))
			{
				list6.Add(item2);
			}
		}
		StringComparer comparer2 = StringComparer.Create(LocManager.Instance.CultureInfo, CompareOptions.None);
		list6.Sort((RelicModel p1, RelicModel p2) => comparer2.Compare(p1.Title.GetFormattedText(), p2.Title.GetFormattedText()));
		LoadRelicNodes(list6.Concat(list5), seenRelics, allUnlockedRelics);
		_collection.AddRelics(list6);
		_collection.AddRelics(list5);
	}

	private void LoadSubcategory(NRelicCollection collection, LocString? header, IEnumerable<RelicModel> relics, HashSet<RelicModel> seenRelics, HashSet<RelicModel> unlockedRelics)
	{
		_headerLabel.Text = header?.GetFormattedText() ?? "";
		_collection = collection;
		_collection.AddRelics(relics);
		LoadRelicNodes(relics, seenRelics, unlockedRelics);
	}

	private void LoadIcon(Texture2D tex)
	{
		_icon.Texture = tex;
		_icon.Visible = true;
	}

	private void LoadRelicNodes(IEnumerable<RelicModel> relics, HashSet<RelicModel> seenRelics, HashSet<RelicModel> unlockedRelics)
	{
		foreach (Node child in _relicsContainer.GetChildren())
		{
			child.QueueFreeSafely();
		}
		foreach (RelicModel relic in relics)
		{
			ModelVisibility visibility = ((!unlockedRelics.Contains(relic)) ? ModelVisibility.Locked : (seenRelics.Contains(relic) ? ModelVisibility.Visible : ModelVisibility.NotSeen));
			NRelicCollectionEntry nRelicCollectionEntry = NRelicCollectionEntry.Create(relic, visibility);
			_relicsContainer.AddChildSafely(nRelicCollectionEntry);
			nRelicCollectionEntry.Connect(NClickableControl.SignalName.Released, Callable.From<NRelicCollectionEntry>(OnRelicEntryPressed));
		}
	}

	public void ClearRelics()
	{
		foreach (Node child in _relicsContainer.GetChildren())
		{
			child.QueueFreeSafely();
		}
		foreach (NRelicCollectionCategory item in GetChildren().OfType<NRelicCollectionCategory>())
		{
			item.QueueFreeSafely();
		}
	}

	private void OnRelicEntryPressed(NRelicCollectionEntry entry)
	{
		NGame.Instance.GetInspectRelicScreen().Open(_collection.Relics, entry.relic);
		_collection.SetLastFocusedRelic(entry);
	}

	public List<IReadOnlyList<Control>> GetGridItems()
	{
		List<IReadOnlyList<Control>> list = new List<IReadOnlyList<Control>>();
		if (_subCategories.Any())
		{
			foreach (NRelicCollectionCategory subCategory in _subCategories)
			{
				list.AddRange(subCategory.GetGridItems());
			}
		}
		else
		{
			for (int i = 0; i < _relicsContainer.GetChildren().Count; i += _relicsContainer.Columns)
			{
				list.Add(_relicsContainer.GetChildren().OfType<Control>().Skip(i)
					.Take(_relicsContainer.Columns)
					.ToList());
			}
		}
		return list;
	}
}
