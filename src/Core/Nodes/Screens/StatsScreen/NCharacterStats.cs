using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.StatsScreen;

public partial class NCharacterStats : Node
{
	private static readonly string _playtimeIconPath = ImageHelper.GetImagePath("atlases/stats_screen_atlas.sprites/stats_clock.tres");

	private static readonly string _winLossIconPath = ImageHelper.GetImagePath("atlases/stats_screen_atlas.sprites/stats_swords.tres");

	private static readonly string _chainIconPath = ImageHelper.GetImagePath("atlases/stats_screen_atlas.sprites/stats_chain.tres");

	private CharacterStats _characterStats;

	private Control _characterIcon;

	private Node _statsContainer;

	private MegaLabel _nameLabel;

	private MegaLabel _unlocksLabel;

	private NStatEntry _playtimeEntry;

	private NStatEntry _winLossEntry;

	private NStatEntry _streakEntry;

	public static string[] AssetPaths => new string[4] { ScenePath, _playtimeIconPath, _winLossIconPath, _chainIconPath };

	private static string ScenePath => SceneHelper.GetScenePath("screens/stats_screen/character_stats");

	public static NCharacterStats Create(CharacterStats characterStats)
	{
		NCharacterStats nCharacterStats = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NCharacterStats>(PackedScene.GenEditState.Disabled);
		nCharacterStats._characterStats = characterStats;
		return nCharacterStats;
	}

	public override void _Ready()
	{
		CharacterModel byId = ModelDb.GetById<CharacterModel>(_characterStats.Id);
		_characterIcon = GetNode<Control>("%CharacterIcon");
		_characterIcon.AddChildSafely(byId.Icon);
		_statsContainer = GetNode<Node>("%StatsContainer");
		_playtimeEntry = CreateSection(_playtimeIconPath);
		_winLossEntry = CreateSection(_winLossIconPath);
		_streakEntry = CreateSection(_chainIconPath);
		_nameLabel = GetNode<MegaLabel>("%NameLabel");
		_unlocksLabel = GetNode<MegaLabel>("%UnlocksLabel");
		_nameLabel.SetTextAutoSize(byId.Title.GetRawText());
		_nameLabel.AddThemeColorOverride(ThemeConstants.Label.fontColor, byId.NameColor);
		LoadStats();
	}

	private void LoadStats()
	{
		_unlocksLabel.Visible = false;
		LocString locString = new LocString("stats_screen", "ENTRY_CHAR_PLAYTIME.top");
		locString.Add("Playtime", TimeFormatting.Format(_characterStats.Playtime));
		_playtimeEntry.SetTopText(locString.GetFormattedText());
		if (_characterStats.FastestWinTime >= 0)
		{
			locString = new LocString("stats_screen", "ENTRY_CHAR_PLAYTIME.bottom");
			locString.Add("FastestWin", TimeFormatting.Format(_characterStats.FastestWinTime));
			_playtimeEntry.SetBottomText(locString.GetFormattedText());
		}
		locString = new LocString("stats_screen", "ENTRY_CHAR_WIN_LOSS.top");
		if (_characterStats.MaxAscension > 0)
		{
			locString.Add("Amount", _characterStats.MaxAscension);
			_winLossEntry.SetTopText("[red]" + locString.GetFormattedText() + "[/red]");
		}
		locString = new LocString("stats_screen", "ENTRY_CHAR_WIN_LOSS.bottom");
		locString.Add("Wins", _characterStats.TotalWins);
		locString.Add("Losses", _characterStats.TotalLosses);
		_winLossEntry.SetBottomText(locString.GetFormattedText());
		locString = new LocString("stats_screen", "ENTRY_CHAR_STREAK.top");
		locString.Add("Amount", _characterStats.CurrentWinStreak);
		_streakEntry.SetTopText(locString.GetFormattedText());
		locString = new LocString("stats_screen", "ENTRY_CHAR_STREAK.bottom");
		locString.Add("Amount", _characterStats.BestWinStreak);
		_streakEntry.SetBottomText(locString.GetFormattedText());
	}

	private NStatEntry CreateSection(string imgUrl)
	{
		NStatEntry nStatEntry = NStatEntry.Create(imgUrl);
		_statsContainer.AddChildSafely(nStatEntry);
		return nStatEntry;
	}
}
