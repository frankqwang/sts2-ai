using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Multiplayer;
using MegaCrit.Sts2.Core.Nodes.Orbs;
using MegaCrit.Sts2.Core.Nodes.Potions;
using MegaCrit.Sts2.Core.Nodes.Relics;
using MegaCrit.Sts2.Core.Nodes.RestSite;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.Bestiary;
using MegaCrit.Sts2.Core.Nodes.Screens.CardLibrary;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;
using MegaCrit.Sts2.Core.Nodes.Screens.CustomRun;
using MegaCrit.Sts2.Core.Nodes.Screens.DailyRun;
using MegaCrit.Sts2.Core.Nodes.Screens.InspectScreens;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.ModdingScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.PotionLab;
using MegaCrit.Sts2.Core.Nodes.Screens.ProfileScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.RelicCollection;
using MegaCrit.Sts2.Core.Nodes.Screens.RunHistoryScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.Settings;
using MegaCrit.Sts2.Core.Nodes.Screens.StatsScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.Timeline;
using MegaCrit.Sts2.Core.Nodes.Vfx;
using MegaCrit.Sts2.Core.Nodes.Vfx.Cards;
using MegaCrit.Sts2.Core.Rewards;

namespace MegaCrit.Sts2.Core.Assets;

public static class AssetSets
{
	public static IReadOnlySet<string> MainMenuEssentials { get; } = new HashSet<string>([..NMainMenu.AssetPaths, ..NTransition.AssetPaths, ..NSettingsScreen.AssetPaths, "res://shaders/hsv.gdshader", "res://shaders/dark_blur.gdshader"]);

	public static IReadOnlySet<string> IntroLogoAssets { get; } = new HashSet<string>([..NTransition.AssetPaths, ..NLogoAnimation.AssetPaths]);

	public static IReadOnlySet<string> CommonAssets { get; } = new HashSet<string>(new IEnumerable<string>[92]
	{
		NActBanner.AssetPaths,
		NActHistoryEntry.AssetPaths,
		NMultiplayerVoteContainer.AssetPaths,
		NAncientMapPoint.AssetPaths,
		NNormalMapPoint.AssetPaths,
		NAncientNameBanner.AssetPaths,
		NBossMapPoint.AssetPaths,
		NInspectCardScreen.AssetPaths,
		NInspectRelicScreen.AssetPaths,
		NSettingsScreen.AssetPaths,
		NCardPileScreen.AssetPaths,
		NCardRewardSelectionScreen.AssetPaths,
		NChooseABundleSelectionScreen.AssetPaths,
		NChooseACardSelectionScreen.AssetPaths,
		NCombatStartBanner.AssetPaths,
		NCreature.AssetPaths,
		NDailyRunLeaderboard.AssetPaths,
		NDeckCardSelectScreen.AssetPaths,
		NDeckEnchantSelectScreen.AssetPaths,
		NDeckHistoryEntry.AssetPaths,
		NDeckUpgradeSelectScreen.AssetPaths,
		NEndTurnButton.AssetPaths,
		NEnemyTurnBanner.AssetPaths,
		NEnergyCounter.AssetPaths,
		NEventOptionButton.AssetPaths,
		NEventRoom.AssetPaths,
		NHandCardHolder.AssetPaths,
		NHoverTipSet.AssetPaths,
		NIntent.AssetPaths,
		NLinkedRewardSet.AssetPaths,
		NMapScreen.AssetPaths,
		NOrb.AssetPaths,
		NOrbManager.AssetPaths,
		NPlayerTurnBanner.AssetPaths,
		NPotion.AssetPaths,
		NPotionHolder.AssetPaths,
		NPotionPopup.AssetPaths,
		NPower.AssetPaths,
		NRelic.AssetPaths,
		NMultiplayerPlayerState.AssetPaths,
		NRestSiteButton.AssetPaths,
		NRestSiteRoom.AssetPaths,
		NRewardButton.AssetPaths,
		NRewardsScreen.AssetPaths,
		NMapPointHistoryEntry.AssetPaths,
		NRun.AssetPaths,
		NSelectedHandCardHolder.AssetPaths,
		NSimpleCardSelectScreen.AssetPaths,
		NSimpleCardsViewScreen.AssetPaths,
		NSpeechBubbleVfx.AssetPaths,
		NStarCounter.AssetPaths,
		NErrorPopup.AssetPaths,
		NTargetingArrow.AssetPaths,
		NThoughtBubbleVfx.AssetPaths,
		NVerticalPopup.AssetPaths,
		NBlockBrokenVfx.AssetPaths,
		NBlockSparkVfx.AssetPaths,
		NCardBundle.AssetPaths,
		NHorizontalLinesVfx.AssetPaths,
		NExhaustVfx.AssetPaths,
		NCardFlyPowerVfx.AssetPaths,
		NCardFlyShuffleVfx.AssetPaths,
		NCardFlyVfx.AssetPaths,
		NCardRareGlow.AssetPaths,
		NCardSmithVfx.AssetPaths,
		NCardTransformVfx.AssetPaths,
		NCardUncommonGlow.AssetPaths,
		NCardUpgradeVfx.AssetPaths,
		NDamageBlockedVfx.AssetPaths,
		NDamageNumVfx.AssetPaths,
		NDeckViewScreen.AssetPaths,
		NDoomVfx.AssetPaths,
		NGainEpochVfx.AssetPaths,
		NGridCardHolder.AssetPaths,
		NHitSparkVfx.AssetPaths,
		NMapCircleVfx.AssetPaths,
		NMapNodeSelectVfx.AssetPaths,
		NMonsterDeathVfx.AssetPaths,
		NPowerAppliedVfx.AssetPaths,
		NPowerFlashVfx.AssetPaths,
		NPowerRemovedVfx.AssetPaths,
		NPowerUpVfx.AssetPaths,
		NPreviewCardHolder.AssetPaths,
		NRelicFlashVfx.AssetPaths,
		NSmokyVignetteVfx.AssetPaths,
		NStunnedVfx.AssetPaths,
		NUiFlashVfx.AssetPaths,
		VfxCmd.AssetPaths,
		ControllerConfig.AllAssetPaths,
		NTransition.AssetPaths,
		Enum.GetValues<RewardType>().SelectMany((RewardType t) => t.GetAssetPaths()),
		TmpSfx.assetPaths
	}.SelectMany((IEnumerable<string> s) => s).Concat(CardMaterialPaths));

	public static IReadOnlySet<string> MainMenuSet { get; } = new HashSet<string>(new IEnumerable<string>[23]
	{
		NCharacterSelectScreen.AssetPaths,
		NMainMenu.AssetPaths,
		NAbandonRunConfirmPopup.AssetPaths,
		NGenericPopup.AssetPaths,
		NMultiplayerWarningPopup.AssetPaths,
		NCard.AssetPaths,
		NMultiplayerLoadGameScreen.AssetPaths,
		NBestiary.AssetPaths,
		NRelicCollection.AssetPaths,
		NPotionLab.AssetPaths,
		NCardLibrary.AssetPaths,
		NRunHistory.AssetPaths,
		NStatsScreen.AssetPaths,
		NTimelineScreen.AssetPaths,
		NDailyRunScreen.AssetPaths,
		NDailyRunLoadScreen.AssetPaths,
		NCustomRunScreen.AssetPaths,
		NCustomRunLoadScreen.AssetPaths,
		NModdingScreen.AssetPaths,
		NBestiaryEntry.AssetPaths,
		NProfileScreen.AssetPaths,
		NAchievementsGrid.AssetPaths,
		ModelDb.AllCharacters.SelectMany((CharacterModel character) => character.AssetPathsCharacterSelect)
	}.SelectMany((IEnumerable<string> s) => s));

	public static IReadOnlySet<string> RunSet { get; set; } = null;

	public static IReadOnlySet<string> Act { get; set; } = null;

	private static IEnumerable<string> CardMaterialPaths => new global::_003C_003Ez__ReadOnlyArray<string>(new string[16]
	{
		"res://materials/cards/banners/card_banner_common_mat.tres", "res://materials/cards/banners/card_banner_uncommon_mat.tres", "res://materials/cards/banners/card_banner_rare_mat.tres", "res://materials/cards/banners/card_banner_curse_mat.tres", "res://materials/cards/banners/card_banner_status_mat.tres", "res://materials/cards/banners/card_banner_event_mat.tres", "res://materials/cards/banners/card_banner_quest_mat.tres", "res://materials/cards/banners/card_banner_ancient_mat.tres", "res://materials/cards/frames/card_frame_red_mat.tres", "res://materials/cards/frames/card_frame_green_mat.tres",
		"res://materials/cards/frames/card_frame_blue_mat.tres", "res://materials/cards/frames/card_frame_pink_mat.tres", "res://materials/cards/frames/card_frame_orange_mat.tres", "res://materials/cards/frames/card_frame_colorless_mat.tres", "res://materials/cards/frames/card_frame_curse_mat.tres", "res://materials/cards/frames/card_frame_quest_mat.tres"
	});
}
