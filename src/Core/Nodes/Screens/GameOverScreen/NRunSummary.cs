using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;

public partial class NRunSummary : Control
{
	private Control _badgeContainer;

	private Control _discoveryContainer;

	private Control _discoveryHeader;

	private NDiscoveredItem _discoveredCards;

	private NDiscoveredItem _discoveredRelics;

	private NDiscoveredItem _discoveredPotions;

	private NDiscoveredItem _discoveredEnemies;

	private NDiscoveredItem _discoveredEpochs;

	private MegaLabel _cardCount;

	private MegaLabel _relicCount;

	private MegaLabel _potionCount;

	private MegaLabel _enemyCount;

	private MegaLabel _epochCount;

	private Tween? _tween;

	private Tween? _waitTween;

	private const int _maxItemsToList = 10;

	public override void _Ready()
	{
		_badgeContainer = GetNode<Control>("%BadgeContainer");
		_discoveryContainer = GetNode<Control>("%DiscoveryContainer");
		_discoveryHeader = GetNode<Control>("%DiscoveryHeader");
		_cardCount = GetNode<MegaLabel>("%CardCount");
		_relicCount = GetNode<MegaLabel>("%RelicCount");
		_potionCount = GetNode<MegaLabel>("%PotionCount");
		_enemyCount = GetNode<MegaLabel>("%EnemyCount");
		_epochCount = GetNode<MegaLabel>("%EpochCount");
		_discoveredCards = _cardCount.GetParent<NDiscoveredItem>();
		_discoveredRelics = _relicCount.GetParent<NDiscoveredItem>();
		_discoveredPotions = _potionCount.GetParent<NDiscoveredItem>();
		_discoveredEnemies = _enemyCount.GetParent<NDiscoveredItem>();
		_discoveredEpochs = _epochCount.GetParent<NDiscoveredItem>();
		_discoveredCards.Visible = false;
		_discoveredRelics.Visible = false;
		_discoveredPotions.Visible = false;
		_discoveredEnemies.Visible = false;
		_discoveredEpochs.Visible = false;
	}

	public async Task AnimateInDiscoveries(RunState runState)
	{
		Player player = LocalContext.GetMe(runState);
		if (player.DiscoveredCards.Count + player.DiscoveredRelics.Count + player.DiscoveredPotions.Count + player.DiscoveredEnemies.Count + player.DiscoveredEpochs.Count == 0)
		{
			Log.Info("No discoveries this time. Very sad");
			return;
		}
		Tween tween = CreateTween();
		tween.TweenProperty(_discoveryHeader, "modulate:a", 1f, 0.25);
		await Task.Delay(100);
		if (player.DiscoveredCards.Count > 0)
		{
			string discoveryBodyText = GetDiscoveryBodyText(player.DiscoveredCards, (ModelId id) => ModelDb.GetById<CardModel>(id).Title, "game_over_screen", "DISCOVERY_BODY_CARD", "CardCount");
			_discoveredCards.SetHoverTip(new HoverTip(new LocString("game_over_screen", "DISCOVERY_HEADER_CARD"), discoveryBodyText));
			_discoveredCards.Visible = true;
			_discoveredCards.Modulate = StsColors.transparentBlack;
		}
		if (player.DiscoveredRelics.Count > 0)
		{
			string discoveryBodyText2 = GetDiscoveryBodyText(player.DiscoveredRelics, (ModelId id) => ModelDb.GetById<RelicModel>(id).Title.GetFormattedText(), "game_over_screen", "DISCOVERY_BODY_RELIC", "RelicCount");
			_discoveredRelics.SetHoverTip(new HoverTip(new LocString("game_over_screen", "DISCOVERY_HEADER_RELIC"), discoveryBodyText2));
			_discoveredRelics.Visible = true;
			_discoveredRelics.Modulate = StsColors.transparentBlack;
		}
		if (player.DiscoveredPotions.Count > 0)
		{
			string discoveryBodyText3 = GetDiscoveryBodyText(player.DiscoveredPotions, (ModelId id) => ModelDb.GetById<PotionModel>(id).Title.GetFormattedText(), "game_over_screen", "DISCOVERY_BODY_POTION", "PotionCount");
			_discoveredPotions.SetHoverTip(new HoverTip(new LocString("game_over_screen", "DISCOVERY_HEADER_POTION"), discoveryBodyText3));
			_discoveredPotions.Visible = true;
			_discoveredPotions.Modulate = StsColors.transparentBlack;
		}
		if (player.DiscoveredEnemies.Count > 0)
		{
			string discoveryBodyText4 = GetDiscoveryBodyText(player.DiscoveredEnemies, (ModelId id) => ModelDb.GetById<MonsterModel>(id).Title.GetFormattedText(), "game_over_screen", "DISCOVERY_BODY_ENEMY", "EnemyCount");
			_discoveredEnemies.SetHoverTip(new HoverTip(new LocString("game_over_screen", "DISCOVERY_HEADER_ENEMY"), discoveryBodyText4));
			_discoveredEnemies.Visible = true;
			_discoveredEnemies.Modulate = StsColors.transparentBlack;
		}
		if (player.DiscoveredEpochs.Count > 0)
		{
			LocString title = new LocString("game_over_screen", "DISCOVERY_HEADER_EPOCH");
			LocString locString = new LocString("game_over_screen", "DISCOVERY_BODY_EPOCH");
			locString.Add("EpochCount", player.DiscoveredEpochs.Count);
			HoverTip hoverTip = new HoverTip(title, locString);
			_discoveredEpochs.SetHoverTip(hoverTip);
			_discoveredEpochs.Visible = true;
			_discoveredEpochs.Modulate = StsColors.transparentBlack;
		}
		if (_discoveredCards.Visible)
		{
			_cardCount.SetTextAutoSize($"{player.DiscoveredCards.Count}");
			await TaskHelper.RunSafely(DiscoveryAnimHelper(_discoveredCards));
		}
		if (_discoveredRelics.Visible)
		{
			_relicCount.SetTextAutoSize($"{player.DiscoveredRelics.Count}");
			await TaskHelper.RunSafely(DiscoveryAnimHelper(_discoveredRelics));
		}
		if (_discoveredPotions.Visible)
		{
			_potionCount.SetTextAutoSize($"{player.DiscoveredPotions.Count}");
			await TaskHelper.RunSafely(DiscoveryAnimHelper(_discoveredPotions));
		}
		if (_discoveredEnemies.Visible)
		{
			_enemyCount.SetTextAutoSize($"{player.DiscoveredEnemies.Count}");
			await TaskHelper.RunSafely(DiscoveryAnimHelper(_discoveredEnemies));
		}
		if (_discoveredEpochs.Visible)
		{
			_epochCount.SetTextAutoSize($"{player.DiscoveredEpochs.Count}");
			await TaskHelper.RunSafely(DiscoveryAnimHelper(_discoveredEpochs));
		}
	}

	private async Task DiscoveryAnimHelper(Control node)
	{
		node.Modulate = StsColors.transparentBlack;
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(node, "modulate", Colors.White, 0.3);
		_tween.TweenProperty(node, "position:y", 0f, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back)
			.From(100f);
		await ToSignal(_tween, Tween.SignalName.Finished);
	}

	private static string GetDiscoveryBodyText<T>(List<T> discoveredIds, Func<T, string> getTitle, string locTable, string locKey, string countParam)
	{
		LocString locString = new LocString(locTable, locKey);
		locString.Add(countParam, discoveredIds.Count);
		string text = string.Join("\n", discoveredIds.Take(10).Select(getTitle));
		if (discoveredIds.Count > 10)
		{
			text += "\n....";
		}
		return locString.GetFormattedText() + "\n\n" + text;
	}
}
