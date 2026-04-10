using System;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Nodes.Screens.Settings;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens.StatsScreen;

public partial class NStatsScreen : NSubmenu
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/stats_screen/stats_screen");

	private NStatsTabManager _statsTabManager;

	private NSettingsTab _statsTab;

	private NSettingsTab _achievementsTab;

	private NGeneralStatsGrid _statsGrid;

	private Tween? _screenTween;

	public static string[] AssetPaths
	{
		get
		{
			string scenePath = _scenePath;
			string[] assetPaths = NGeneralStatsGrid.AssetPaths;
			int num = 0;
			string[] array = new string[1 + assetPaths.Length];
			array[num] = scenePath;
			num++;
			ReadOnlySpan<string> readOnlySpan = new ReadOnlySpan<string>(assetPaths);
			readOnlySpan.CopyTo(new Span<string>(array).Slice(num, readOnlySpan.Length));
			num += readOnlySpan.Length;
			return array;
		}
	}

	protected override Control InitialFocusedControl => _statsGrid.DefaultFocusedControl;

	public static NStatsScreen? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NStatsScreen>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		ConnectSignals();
		_statsTab = GetNode<NSettingsTab>("%StatsTab");
		_statsTab.SetLabel(new LocString("stats_screen", "TAB_STATS.header").GetFormattedText());
		_achievementsTab = GetNode<NSettingsTab>("%Achievements");
		_achievementsTab.SetLabel(new LocString("stats_screen", "TAB_ACHIEVEMENT.header").GetFormattedText());
		_statsTab.Connect(NClickableControl.SignalName.Released, Callable.From<NClickableControl>(delegate
		{
			OpenStatsMenu();
		}));
		_statsTabManager = GetNode<NStatsTabManager>("%Tabs");
		_statsGrid = GetNode<NGeneralStatsGrid>("%StatsGrid");
		GetNode<MegaLabel>("%OverallStatsHeader").SetTextAutoSize(new LocString("main_menu_ui", "STATISTICS.OVERALL.title").GetFormattedText());
		GetNode<MegaLabel>("%CharacterStatsHeader").SetTextAutoSize(new LocString("main_menu_ui", "STATISTICS.title").GetFormattedText());
		_achievementsTab.Disable();
	}

	public override void OnSubmenuOpened()
	{
		_screenTween?.Kill();
		_screenTween = CreateTween();
		_screenTween.TweenProperty(this, "modulate:a", 1f, 0.4).From(0f);
		base.Visible = true;
		OpenStatsMenu();
		_statsTabManager.ResetTabs();
	}

	private void OpenStatsMenu()
	{
		_statsGrid.Visible = true;
		_statsGrid.LoadStats();
		ActiveScreenContext.Instance.Update();
	}

	private void OpenAchievementsMenu()
	{
		_statsGrid.Visible = false;
		ActiveScreenContext.Instance.Update();
	}
}
