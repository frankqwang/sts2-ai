using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.Bestiary;
using MegaCrit.Sts2.Core.Nodes.Screens.CardLibrary;
using MegaCrit.Sts2.Core.Nodes.Screens.PotionLab;
using MegaCrit.Sts2.Core.Nodes.Screens.RelicCollection;
using MegaCrit.Sts2.Core.Nodes.Screens.RunHistoryScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.StatsScreen;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;

public partial class NCompendiumSubmenu : NSubmenu
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/compendium_submenu");

	private NButton _confirmButton;

	private NShortSubmenuButton _cardLibraryButton;

	private NShortSubmenuButton _relicCollectionButton;

	private NShortSubmenuButton _potionLabButton;

	private NShortSubmenuButton _bestiaryButton;

	private NCompendiumBottomButton _leaderboardsButton;

	private NCompendiumBottomButton _statisticsButton;

	private NCompendiumBottomButton _runHistoryButton;

	private IRunState _runState;

	protected override Control InitialFocusedControl => _cardLibraryButton;

	public static NCompendiumSubmenu? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NCompendiumSubmenu>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		ConnectSignals();
		_cardLibraryButton = GetNode<NShortSubmenuButton>("%CardLibraryButton");
		_cardLibraryButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OpenCardLibrary));
		_cardLibraryButton.SetIconAndLocalization("COMPENDIUM_CARD_LIBRARY");
		_relicCollectionButton = GetNode<NShortSubmenuButton>("%RelicCollectionButton");
		_relicCollectionButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OpenRelicCollection));
		_relicCollectionButton.SetIconAndLocalization("COMPENDIUM_RELIC_COLLECTION");
		_potionLabButton = GetNode<NShortSubmenuButton>("%PotionLabButton");
		_potionLabButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OpenPotionLab));
		_potionLabButton.SetIconAndLocalization("COMPENDIUM_POTION_LAB");
		_bestiaryButton = GetNode<NShortSubmenuButton>("%BestiaryButton");
		_bestiaryButton.Disable();
		_bestiaryButton.SetIconAndLocalization("COMPENDIUM_BESTIARY");
		_leaderboardsButton = GetNode<NCompendiumBottomButton>("%LeaderboardsButton");
		_leaderboardsButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OpenLeaderboards));
		_leaderboardsButton.SetLocalization("LEADERBOARDS");
		_statisticsButton = GetNode<NCompendiumBottomButton>("%StatisticsButton");
		_statisticsButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OpenStatistics));
		_statisticsButton.SetLocalization("STATISTICS");
		_runHistoryButton = GetNode<NCompendiumBottomButton>("%RunHistoryButton");
		_runHistoryButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OpenRunHistory));
		_runHistoryButton.SetLocalization("RUN_HISTORY");
		int num = 4;
		List<Control> list = new List<Control>(num);
		CollectionsMarshal.SetCount(list, num);
		Span<Control> span = CollectionsMarshal.AsSpan(list);
		int num2 = 0;
		span[num2] = _cardLibraryButton;
		num2++;
		span[num2] = _relicCollectionButton;
		num2++;
		span[num2] = _potionLabButton;
		num2++;
		span[num2] = _bestiaryButton;
		List<Control> list2 = list;
		num2 = 3;
		List<Control> list3 = new List<Control>(num2);
		CollectionsMarshal.SetCount(list3, num2);
		span = CollectionsMarshal.AsSpan(list3);
		num = 0;
		span[num] = _leaderboardsButton;
		num++;
		span[num] = _statisticsButton;
		num++;
		span[num] = _runHistoryButton;
		List<Control> list4 = list3;
		for (int i = 0; i < list2.Count; i++)
		{
			list2[i].FocusNeighborTop = list2[i].GetPath();
			list2[i].FocusNeighborLeft = ((i > 0) ? list2[i - 1].GetPath() : list2[i].GetPath());
			list2[i].FocusNeighborRight = ((i < list2.Count - 1) ? list2[i + 1].GetPath() : list2[i].GetPath());
		}
		for (int j = 0; j < list4.Count; j++)
		{
			list4[j].FocusNeighborBottom = list4[j].GetPath();
			list4[j].FocusNeighborLeft = ((j > 0) ? list4[j - 1].GetPath() : list4[j].GetPath());
			list4[j].FocusNeighborRight = ((j < list4.Count - 1) ? list4[j + 1].GetPath() : list4[j].GetPath());
		}
		list2[0].FocusNeighborBottom = list4[0].GetPath();
		list2[1].FocusNeighborBottom = list4[0].GetPath();
		list2[2].FocusNeighborBottom = list4[1].GetPath();
		list2[3].FocusNeighborBottom = list4[2].GetPath();
		list4[0].FocusNeighborTop = list2[1].GetPath();
		list4[1].FocusNeighborTop = list2[2].GetPath();
		list4[2].FocusNeighborTop = list2[3].GetPath();
	}

	public override void OnSubmenuOpened()
	{
		base.OnSubmenuOpened();
		_leaderboardsButton.Visible = false;
		_runHistoryButton.Visible = NRunHistory.CanBeShown();
	}

	public void Initialize(IRunState runState)
	{
		_runState = runState;
	}

	private void OpenCardLibrary(NButton _)
	{
		NCardLibrary submenuType = _stack.GetSubmenuType<NCardLibrary>();
		submenuType.Initialize(_runState);
		_stack.Push(submenuType);
	}

	private void OpenRelicCollection(NButton _)
	{
		_stack.PushSubmenuType<NRelicCollection>();
	}

	private void OpenPotionLab(NButton _)
	{
		_stack.PushSubmenuType<NPotionLab>();
	}

	private void OpenBestiary(NButton _)
	{
		_stack.PushSubmenuType<NBestiary>();
	}

	private void OpenLeaderboards(NButton _)
	{
	}

	private void OpenStatistics(NButton _)
	{
		_stack.PushSubmenuType<NStatsScreen>();
	}

	private void OpenRunHistory(NButton _)
	{
		_stack.PushSubmenuType<NRunHistory>();
	}
}
