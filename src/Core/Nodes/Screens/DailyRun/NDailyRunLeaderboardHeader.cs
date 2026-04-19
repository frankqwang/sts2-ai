using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens.DailyRun;

public partial class NDailyRunLeaderboardHeader : Control
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/daily_run/daily_run_leaderboard_header");

	private MegaLabel _rank;

	private MegaLabel _name;

	private MegaLabel _score;

	public static NDailyRunLeaderboardHeader? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NDailyRunLeaderboardHeader>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		_name = GetNode<MegaLabel>("%Name");
		_name.SetTextAutoSize(new LocString("main_menu_ui", "LEADERBOARDS.nameHeader").GetRawText());
	}
}
