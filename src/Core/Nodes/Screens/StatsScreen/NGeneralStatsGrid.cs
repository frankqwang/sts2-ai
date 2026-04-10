using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Characters;
using MegaCrit.Sts2.Core.Platform;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Timeline;

namespace MegaCrit.Sts2.Core.Nodes.Screens.StatsScreen;

public partial class NGeneralStatsGrid : Control
{
	private static readonly string _achievementsIconPath = ImageHelper.GetImagePath("atlases/stats_screen_atlas.sprites/stats_achievements.tres");

	private static readonly string _playtimeIconPath = ImageHelper.GetImagePath("atlases/stats_screen_atlas.sprites/stats_clock.tres");

	private static readonly string _cardsIconPath = ImageHelper.GetImagePath("atlases/stats_screen_atlas.sprites/stats_cards.tres");

	private static readonly string _winLossIconPath = ImageHelper.GetImagePath("atlases/stats_screen_atlas.sprites/stats_swords.tres");

	private static readonly string _monsterIconPath = ImageHelper.GetImagePath("atlases/stats_screen_atlas.sprites/stats_monsters.tres");

	private static readonly string _ancientsIconPath = ImageHelper.GetImagePath("atlases/stats_screen_atlas.sprites/stats_ancients.tres");

	private static readonly string _relicIconPath = ImageHelper.GetImagePath("atlases/stats_screen_atlas.sprites/stats_chest.tres");

	private static readonly string _potionIconPath = ImageHelper.GetImagePath("atlases/stats_screen_atlas.sprites/stats_potions_seen.tres");

	private static readonly string _eventsIconPath = ImageHelper.GetImagePath("atlases/stats_screen_atlas.sprites/stats_questionmark.tres");

	private static readonly string _streakIconPath = ImageHelper.GetImagePath("atlases/stats_screen_atlas.sprites/stats_chain.tres");

	private Node _gridContainer;

	private NStatEntry _achievementsEntry;

	private NStatEntry _playtimeEntry;

	private NStatEntry _cardsEntry;

	private NStatEntry _winLossEntry;

	private NStatEntry _monsterEntry;

	private NStatEntry _relicEntry;

	private NStatEntry _potionEntry;

	private NStatEntry _eventsEntry;

	private NStatEntry _streakEntry;

	private Control _characterStatContainer;

	private Tween? _screenTween;

	public static string[] AssetPaths
	{
		get
		{
			List<string> list = new List<string>();
			list.Add(_achievementsIconPath);
			list.Add(_playtimeIconPath);
			list.Add(_cardsIconPath);
			list.Add(_winLossIconPath);
			list.Add(_monsterIconPath);
			list.Add(_ancientsIconPath);
			list.Add(_relicIconPath);
			list.Add(_potionIconPath);
			list.Add(_eventsIconPath);
			list.Add(_streakIconPath);
			list.AddRange(NCharacterStats.AssetPaths);
			return list.ToArray();
		}
	}

	private static HoverTip PlaytimeTip => new HoverTip(new LocString("stats_screen", "TIP_PLAYTIME.header"), new LocString("stats_screen", "TIP_PLAYTIME.description"));

	private static HoverTip WinsLossesTip => new HoverTip(new LocString("stats_screen", "TIP_WIN_LOSS.header"), new LocString("stats_screen", "TIP_WIN_LOSS.description"));

	public Control DefaultFocusedControl => _achievementsEntry;

	public override void _Ready()
	{
		_gridContainer = GetNode<Control>("%GridContainer");
		_characterStatContainer = GetNode<Control>("%CharacterStatsContainer");
		_achievementsEntry = CreateSection(_achievementsIconPath);
		_playtimeEntry = CreateSection(_playtimeIconPath);
		_cardsEntry = CreateSection(_cardsIconPath);
		_winLossEntry = CreateSection(_winLossIconPath);
		_monsterEntry = CreateSection(_monsterIconPath);
		_relicEntry = CreateSection(_relicIconPath);
		_potionEntry = CreateSection(_potionIconPath);
		_eventsEntry = CreateSection(_eventsIconPath);
		_streakEntry = CreateSection(_streakIconPath);
		SetupHoverTips();
	}

	private NStatEntry CreateSection(string imgUrl)
	{
		NStatEntry nStatEntry = NStatEntry.Create(imgUrl);
		_gridContainer.AddChildSafely(nStatEntry);
		return nStatEntry;
	}

	public void LoadStats()
	{
		ProgressState progressSave = SaveManager.Instance.Progress;
		SaveManager instance = SaveManager.Instance;
		LocString locString = new LocString("stats_screen", "ENTRY_ACHIEVEMENTS.top");
		int denominator = AchievementsUtil.TotalAchievementCount();
		int numerator = AchievementsUtil.UnlockedAchievementCount();
		locString.Add("Amount", StringHelper.RatioFormat(numerator, denominator));
		_achievementsEntry.SetTopText(locString.GetFormattedText() ?? "");
		locString = new LocString("stats_screen", "ENTRY_ACHIEVEMENTS.bottom");
		int numerator2 = progressSave.Epochs.Count((SerializableEpoch epoch) => epoch.State >= EpochState.Revealed);
		if (EpochModel.AllEpochIds.All((string id) => progressSave.Epochs.Any((SerializableEpoch epoch) => epoch.Id == id)))
		{
			int count = progressSave.Epochs.Count;
			locString.Add("Amount", StringHelper.RatioFormat(numerator2, count));
		}
		else
		{
			locString.Add("Amount", StringHelper.RatioFormat(numerator2.ToString(), "??"));
		}
		_achievementsEntry.SetBottomText(locString.GetFormattedText() ?? "");
		locString = new LocString("stats_screen", "ENTRY_PLAYTIME.top");
		locString.Add("Playtime", TimeFormatting.Format(progressSave.TotalPlaytime));
		_playtimeEntry.SetTopText(locString.GetFormattedText());
		if (progressSave.Wins > 0)
		{
			locString = new LocString("stats_screen", "ENTRY_PLAYTIME.bottom");
			locString.Add("FastestWin", TimeFormatting.Format(progressSave.FastestVictory));
			_playtimeEntry.SetBottomText(locString.GetFormattedText());
		}
		locString = new LocString("stats_screen", "ENTRY_CARDS.top");
		locString.Add("Amount", StringHelper.RatioFormat(instance.GetTotalUnlockedCards(), SaveManager.GetUnlockableCardCount()));
		_cardsEntry.SetTopText(locString.GetFormattedText() ?? "");
		locString = new LocString("stats_screen", "ENTRY_CARDS.bottom");
		locString.Add("Amount", StringHelper.RatioFormat(progressSave.DiscoveredCards.Count, ModelDb.AllCards.Count()));
		_cardsEntry.SetBottomText(locString.GetFormattedText());
		int aggregateAscensionProgress = instance.GetAggregateAscensionProgress();
		if (aggregateAscensionProgress > 0)
		{
			locString = new LocString("stats_screen", "ENTRY_WIN_LOSS.top");
			locString.Add("Amount", StringHelper.RatioFormat(aggregateAscensionProgress, SaveManager.GetAggregateAscensionCount()));
			_winLossEntry.SetTopText(locString.GetFormattedText() ?? "");
		}
		locString = new LocString("stats_screen", "ENTRY_WIN_LOSS.bottom");
		locString.Add("Wins", progressSave.Wins);
		locString.Add("Losses", progressSave.Losses);
		_winLossEntry.SetBottomText(locString.GetFormattedText());
		locString = new LocString("stats_screen", "ENTRY_MONSTER.top");
		locString.Add("Amount", StringHelper.Radix(instance.GetTotalKills()));
		_monsterEntry.SetTopText(locString.GetFormattedText() ?? "");
		locString = new LocString("stats_screen", "ENTRY_MONSTER.bottom");
		locString.Add("Amount", StringHelper.RatioFormat(instance.Progress.EnemyStats.Count, ModelDb.Monsters.Count()));
		_monsterEntry.SetBottomText(locString.GetFormattedText() ?? "");
		locString = new LocString("stats_screen", "ENTRY_RELIC.top");
		locString.Add("Amount", StringHelper.RatioFormat(instance.GetTotalUnlockedRelics(), SaveManager.GetUnlockableRelicCount()));
		_relicEntry.SetTopText(locString.GetFormattedText() ?? "");
		locString = new LocString("stats_screen", "ENTRY_RELIC.bottom");
		locString.Add("Amount", StringHelper.RatioFormat(progressSave.DiscoveredRelics.Count, ModelDb.AllRelics.Count()));
		_relicEntry.SetBottomText(locString.GetFormattedText());
		locString = new LocString("stats_screen", "ENTRY_POTION.top");
		locString.Add("Amount", StringHelper.RatioFormat(instance.GetTotalUnlockedPotions(), SaveManager.GetUnlockablePotionCount()));
		_potionEntry.SetTopText(locString.GetFormattedText() ?? "");
		locString = new LocString("stats_screen", "ENTRY_POTION.bottom");
		locString.Add("Amount", ModelDb.AllPotions.Count());
		_potionEntry.SetBottomText(locString.GetFormattedText());
		locString = new LocString("stats_screen", "ENTRY_EVENTS.top");
		locString.Add("Amount", "N/A");
		_eventsEntry.SetTopText(locString.GetFormattedText());
		locString = new LocString("stats_screen", "ENTRY_EVENTS.bottom");
		locString.Add("Amount", StringHelper.RatioFormat(progressSave.DiscoveredEvents.Count, ModelDb.AllEvents.Count()));
		_eventsEntry.SetBottomText(locString.GetFormattedText());
		locString = new LocString("stats_screen", "ENTRY_STREAK.top");
		locString.Add("Amount", progressSave.BestWinStreak);
		_streakEntry.SetTopText(locString.GetFormattedText());
		if (aggregateAscensionProgress > 99999999)
		{
			locString = new LocString("stats_screen", "ENTRY_STREAK.bottom");
			locString.Add("Amount", 5m);
			_streakEntry.SetBottomText("[red]" + locString.GetFormattedText() + "[/red]");
		}
		_characterStatContainer.FreeChildren();
		CreateCharacterSection(progressSave, ModelDb.Character<Ironclad>().Id);
		CreateCharacterSection(progressSave, ModelDb.Character<Silent>().Id);
		CreateCharacterSection(progressSave, ModelDb.Character<Regent>().Id);
		CreateCharacterSection(progressSave, ModelDb.Character<Necrobinder>().Id);
		CreateCharacterSection(progressSave, ModelDb.Character<Defect>().Id);
	}

	private void CreateCharacterSection(ProgressState progressSave, ModelId id)
	{
		CharacterStats statsForCharacter = progressSave.GetStatsForCharacter(id);
		if (statsForCharacter != null)
		{
			NCharacterStats child = NCharacterStats.Create(statsForCharacter);
			_characterStatContainer.AddChildSafely(child);
		}
	}

	private void SetupHoverTips()
	{
		_playtimeEntry.SetHoverTip(PlaytimeTip);
		int aggregateAscensionProgress = SaveManager.Instance.GetAggregateAscensionProgress();
		if (aggregateAscensionProgress > 0)
		{
			_winLossEntry.SetHoverTip(WinsLossesTip);
		}
	}
}
