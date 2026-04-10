using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Characters;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Potions;
using MegaCrit.Sts2.Core.Nodes.Relics;
using MegaCrit.Sts2.Core.Nodes.Screens.Capstones;
using MegaCrit.Sts2.Core.Nodes.Screens.RunHistoryScreen;
using MegaCrit.Sts2.Core.Nodes.Vfx;
using MegaCrit.Sts2.Core.Platform;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Multiplayer;

public partial class NMultiplayerPlayerState : Control
{
	private const ulong _delayBetweenTweensMsec = 500uL;

	private static readonly string _scenePath = SceneHelper.GetScenePath("ui/multiplayer_player_state");

	private static readonly string _cardScenePath = SceneHelper.GetScenePath("screens/run_history_screen/deck_history_entry");

	private const string _darkenedEnergyMatPath = "res://materials/ui/energy_orb_dark.tres";

	private const float _refHpBarWidth = 175f;

	private const float _refHpBarMaxHp = 80f;

	private const float _selectionReticlePadding = 6f;

	private NHealthBar _healthBar;

	private TextureRect _characterIcon;

	private MegaLabel _nameplateLabel;

	private HBoxContainer _topContainer;

	private TextureRect _turnEndIndicator;

	private TextureRect _disconnectedIndicator;

	private NMultiplayerNetworkProblemIndicator _networkProblemIndicator;

	private NSelectionReticle _selectionReticle;

	private TextureRect _locationIcon;

	private Control _locationContainer;

	private Control _energyContainer;

	private TextureRect _energyImage;

	private MegaLabel _energyCount;

	private Control _starContainer;

	private MegaLabel _starCount;

	private Control _cardContainer;

	private NTinyCard _cardImage;

	private MegaLabel _cardCount;

	private Tween? _locationIconTween;

	private bool _isMouseOver;

	private bool _isCreatureHovered;

	private bool _isHighlighted;

	private bool _focusedWhileTargeting;

	private ulong _nextTweenTime;

	private Texture2D? _currentLocationIcon;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlyArray<string>(new string[2] { _scenePath, _cardScenePath });

	public NButton Hitbox { get; private set; }

	public Player Player { get; private set; }

	public static NMultiplayerPlayerState Create(Player player)
	{
		NMultiplayerPlayerState nMultiplayerPlayerState = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NMultiplayerPlayerState>(PackedScene.GenEditState.Disabled);
		nMultiplayerPlayerState.Player = player;
		return nMultiplayerPlayerState;
	}

	public override void _Ready()
	{
		_nameplateLabel = GetNode<MegaLabel>("%NameplateLabel");
		_healthBar = GetNode<NHealthBar>("%HealthBar");
		_characterIcon = GetNode<TextureRect>("%CharacterIcon");
		_turnEndIndicator = GetNode<TextureRect>("%TurnEndIndicator");
		_disconnectedIndicator = GetNode<TextureRect>("%DisconnectedIndicator");
		_networkProblemIndicator = GetNode<NMultiplayerNetworkProblemIndicator>("%NetworkProblemIndicator");
		_selectionReticle = GetNode<NSelectionReticle>("%SelectionReticle");
		Hitbox = GetNode<NButton>("%Hitbox");
		_locationIcon = GetNode<TextureRect>("%LocationIcon");
		_locationContainer = GetNode<Control>("%LocationContainer");
		_topContainer = GetNode<HBoxContainer>("TopInfoContainer");
		_energyContainer = GetNode<Control>("%EnergyCountContainer");
		_energyImage = _energyContainer.GetNode<TextureRect>("Image");
		_energyCount = _energyContainer.GetNode<MegaLabel>("EnergyCount");
		_starContainer = GetNode<Control>("%StarCountContainer");
		_starCount = _starContainer.GetNode<MegaLabel>("StarCount");
		_cardContainer = GetNode<Control>("%CardCountContainer");
		_cardImage = _cardContainer.GetNode<NTinyCard>("TinyCard");
		_cardCount = _cardContainer.GetNode<MegaLabel>("CardCount");
		_selectionReticle.Visible = true;
		_characterIcon.Texture = Player.Character.IconTexture;
		_nameplateLabel.SetTextAutoSize(PlatformUtil.GetPlayerName(RunManager.Instance.NetService.Platform, Player.NetId));
		_healthBar.SetCreature(Player.Creature);
		_networkProblemIndicator.Initialize(Player.NetId);
		_locationContainer.Visible = false;
		_energyContainer.Visible = false;
		_energyImage.Texture = ResourceLoader.Load<Texture2D>(Player.Character.CardPool.EnergyIconPath, null, ResourceLoader.CacheMode.Reuse);
		_starContainer.Visible = false;
		_cardContainer.Visible = false;
		_cardImage.Set(Player.Character.CardPool, CardType.Attack, CardRarity.Common);
		_turnEndIndicator.Visible = false;
		_healthBar.FadeOutHpLabel(0f, 0f);
		Player.Creature.BlockChanged += BlockChanged;
		Player.Creature.CurrentHpChanged += OnCreatureValueChanged;
		Player.Creature.MaxHpChanged += OnCreatureValueChanged;
		Player.Creature.PowerApplied += OnPowerAppliedOrRemoved;
		Player.Creature.PowerIncreased += OnPowerIncreased;
		Player.Creature.PowerDecreased += OnPowerDecreased;
		Player.Creature.PowerRemoved += OnPowerAppliedOrRemoved;
		Player.Creature.Died += OnCreatureChanged;
		Player.RelicObtained += OnRelicObtained;
		Player.RelicRemoved += OnRelicRemoved;
		Player.PotionProcured += OnPotionProcured;
		Player.PotionDiscarded += OnPotionDiscarded;
		Player.Deck.CardAdded += OnCardObtained;
		Player.Deck.CardRemoved += OnCardRemovedFromDeck;
		CombatManager.Instance.PlayerEndedTurn += RefreshPlayerReadyIndicator;
		CombatManager.Instance.PlayerUnendedTurn += RefreshPlayerReadyIndicator;
		CombatManager.Instance.TurnStarted += OnTurnStarted;
		CombatManager.Instance.CombatSetUp += OnCombatSetUp;
		CombatManager.Instance.CombatEnded += OnCombatEnded;
		RunManager.Instance.FlavorSynchronizer.OnEndTurnPingReceived += OnPlayerEndTurnPing;
		RunManager.Instance.InputSynchronizer.ScreenChanged += OnPlayerScreenChanged;
		RunManager.Instance.MapSelectionSynchronizer.PlayerVoteChanged += OnPlayerVoteChanged;
		RunManager.Instance.MapSelectionSynchronizer.PlayerVoteCancelled += RefreshPlayerReadyIndicator;
		RunManager.Instance.MapSelectionSynchronizer.PlayerVotesCleared += OnPlayerVotesCleared;
		if (RunManager.Instance.RunLobby != null)
		{
			RunManager.Instance.RunLobby.RemotePlayerDisconnected += RefreshConnectedState;
			RunManager.Instance.RunLobby.LocalPlayerDisconnected += RefreshConnectedState;
			RunManager.Instance.RunLobby.PlayerRejoined += RefreshConnectedState;
		}
		Hitbox.Connect(NClickableControl.SignalName.Focused, Callable.From<NButton>(OnFocus));
		Hitbox.Connect(NClickableControl.SignalName.Unfocused, Callable.From<NButton>(OnUnfocus));
		Hitbox.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnRelease));
		RefreshValues();
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		Player.Creature.BlockChanged -= BlockChanged;
		Player.Creature.CurrentHpChanged -= OnCreatureValueChanged;
		Player.Creature.MaxHpChanged -= OnCreatureValueChanged;
		Player.Creature.PowerApplied -= OnPowerAppliedOrRemoved;
		Player.Creature.PowerIncreased -= OnPowerIncreased;
		Player.Creature.PowerDecreased -= OnPowerDecreased;
		Player.Creature.PowerRemoved -= OnPowerAppliedOrRemoved;
		Player.Creature.Died -= OnCreatureChanged;
		Player.RelicObtained -= OnRelicObtained;
		Player.RelicRemoved -= OnRelicRemoved;
		Player.PotionProcured -= OnPotionProcured;
		Player.PotionDiscarded -= OnPotionDiscarded;
		Player.Deck.CardAdded -= OnCardObtained;
		Player.Deck.CardRemoved -= OnCardRemovedFromDeck;
		CombatManager.Instance.PlayerEndedTurn -= RefreshPlayerReadyIndicator;
		CombatManager.Instance.PlayerUnendedTurn -= RefreshPlayerReadyIndicator;
		CombatManager.Instance.TurnStarted -= OnTurnStarted;
		CombatManager.Instance.CombatSetUp -= OnCombatSetUp;
		CombatManager.Instance.CombatEnded -= OnCombatEnded;
		RunManager.Instance.FlavorSynchronizer.OnEndTurnPingReceived -= OnPlayerEndTurnPing;
		RunManager.Instance.InputSynchronizer.ScreenChanged -= OnPlayerScreenChanged;
		RunManager.Instance.MapSelectionSynchronizer.PlayerVoteChanged -= OnPlayerVoteChanged;
		RunManager.Instance.MapSelectionSynchronizer.PlayerVoteCancelled -= RefreshPlayerReadyIndicator;
		RunManager.Instance.MapSelectionSynchronizer.PlayerVotesCleared -= OnPlayerVotesCleared;
		if (RunManager.Instance.RunLobby != null)
		{
			RunManager.Instance.RunLobby.RemotePlayerDisconnected -= RefreshConnectedState;
			RunManager.Instance.RunLobby.LocalPlayerDisconnected -= RefreshConnectedState;
			RunManager.Instance.RunLobby.PlayerRejoined -= RefreshConnectedState;
		}
	}

	private void OnCombatSetUp(CombatState _)
	{
		if (!LocalContext.IsMe(Player))
		{
			_energyContainer.Visible = true;
			Control starContainer = _starContainer;
			int visible;
			if (!(Player.Character is Regent))
			{
				PlayerCombatState? playerCombatState = Player.PlayerCombatState;
				visible = ((playerCombatState != null && playerCombatState.Stars > 0) ? 1 : 0);
			}
			else
			{
				visible = 1;
			}
			starContainer.Visible = (byte)visible != 0;
			_cardContainer.Visible = true;
			Player.PlayerCombatState.EnergyChanged += OnEnergyChanged;
			Player.PlayerCombatState.StarsChanged += OnStarsChanged;
			Player.PlayerCombatState.Hand.CardAdded += OnCardAdded;
			Player.PlayerCombatState.Hand.CardRemoved += OnCardRemoved;
		}
	}

	private void OnCombatEnded(CombatRoom _)
	{
		_turnEndIndicator.Visible = false;
		if (!LocalContext.IsMe(Player))
		{
			_energyContainer.Visible = false;
			_starContainer.Visible = false;
			_cardContainer.Visible = false;
			if (Player.PlayerCombatState != null)
			{
				Player.PlayerCombatState.EnergyChanged -= OnEnergyChanged;
				Player.PlayerCombatState.StarsChanged -= OnStarsChanged;
				Player.PlayerCombatState.Hand.CardAdded -= OnCardAdded;
				Player.PlayerCombatState.Hand.CardRemoved -= OnCardRemoved;
			}
		}
	}

	private void OnCreatureValueChanged(int _, int __)
	{
		RefreshValues();
	}

	private void OnCreatureChanged(Creature _)
	{
		RefreshValues();
	}

	private void OnPowerAppliedOrRemoved(PowerModel _)
	{
		RefreshValues();
	}

	private void OnPowerDecreased(PowerModel _, bool __)
	{
		RefreshValues();
	}

	private void OnPowerIncreased(PowerModel _, int __, bool ___)
	{
		RefreshValues();
	}

	private void RefreshValues()
	{
		UpdateHealthBarWidth();
		_healthBar.RefreshValues();
	}

	private void UpdateHealthBarWidth()
	{
		_healthBar.UpdateWidthRelativeToReferenceValue(80f, 175f);
	}

	private void UpdateSelectionReticleWidth()
	{
		Control control = null;
		foreach (Control item in _topContainer.GetChildren().OfType<Control>())
		{
			if (item.Visible && item.Size.X > 0f)
			{
				control = item;
			}
		}
		float a = control.GlobalPosition.X - base.GlobalPosition.X + control.Size.X + 6f;
		float b = _healthBar.HpBarContainer.GlobalPosition.X - base.GlobalPosition.X + _healthBar.HpBarContainer.Size.X + 6f;
		float x = Mathf.Max(a, b);
		NSelectionReticle selectionReticle = _selectionReticle;
		StringName size = Control.PropertyName.Size;
		Vector2 size2 = _selectionReticle.Size;
		size2.X = x;
		selectionReticle.SetDeferred(size, size2);
		Hitbox.SetDeferred(Control.PropertyName.Size, new Vector2(x, base.Size.Y));
	}

	private void OnEnergyChanged(int _, int __)
	{
		RefreshCombatValues();
	}

	private void OnStarsChanged(int _, int __)
	{
		RefreshCombatValues();
	}

	private void OnCardAdded(CardModel _)
	{
		RefreshCombatValues();
	}

	private void OnCardRemoved(CardModel _)
	{
		RefreshCombatValues();
	}

	private void RefreshCombatValues()
	{
		Control starContainer = _starContainer;
		int visible;
		if (!(Player.Character is Regent))
		{
			PlayerCombatState? playerCombatState = Player.PlayerCombatState;
			visible = ((playerCombatState != null && playerCombatState.Stars > 0) ? 1 : 0);
		}
		else
		{
			visible = 1;
		}
		starContainer.Visible = (byte)visible != 0;
		_energyCount.SetTextAutoSize(Player.PlayerCombatState.Energy.ToString());
		_starCount.SetTextAutoSize(Player.PlayerCombatState.Stars.ToString());
		_cardCount.SetTextAutoSize(Player.PlayerCombatState.Hand.Cards.Count.ToString());
		_energyCount.AddThemeColorOverride(ThemeConstants.Label.fontColor, (Player.PlayerCombatState.Energy == 0) ? StsColors.red : StsColors.cream);
		_energyCount.AddThemeColorOverride(ThemeConstants.Label.fontOutlineColor, (Player.PlayerCombatState.Energy == 0) ? StsColors.unplayableEnergyCostOutline : Player.Character.EnergyLabelOutlineColor);
		Material material = ((Player.PlayerCombatState.Energy == 0) ? PreloadManager.Cache.GetMaterial("res://materials/ui/energy_orb_dark.tres") : null);
		_energyImage.Material = material;
		_energyImage.Modulate = ((Player.PlayerCombatState.Energy == 0) ? Colors.DarkGray : Colors.White);
	}

	public void OnCreatureHovered()
	{
		_isCreatureHovered = true;
		UpdateHighlightedState();
	}

	public void OnCreatureUnhovered()
	{
		_isCreatureHovered = false;
		UpdateHighlightedState();
	}

	public void FlashPlayerReady()
	{
		FlashEndTurn();
	}

	private void UpdateHighlightedState()
	{
		bool flag = _isMouseOver || _isCreatureHovered;
		if (NTargetManager.Instance.IsInSelection && !NTargetManager.Instance.AllowedToTargetNode(this))
		{
			flag = false;
		}
		if (!NTargetManager.Instance.IsInSelection)
		{
			NPlayerHand? instance = NPlayerHand.Instance;
			if (instance != null && instance.InCardPlay)
			{
				flag = false;
			}
		}
		if (_isHighlighted == flag)
		{
			return;
		}
		_isHighlighted = flag;
		if (_isHighlighted)
		{
			_healthBar.FadeInHpLabel(0.1f);
			UpdateSelectionReticleWidth();
			_selectionReticle.OnSelect();
			if (_networkProblemIndicator.IsShown)
			{
				LocString locString;
				LocString title;
				if (RunManager.Instance.NetService.Type == NetGameType.Client)
				{
					locString = new LocString("static_hover_tips", "NETWORK_PROBLEM_CLIENT.description");
					title = new LocString("static_hover_tips", "NETWORK_PROBLEM_CLIENT.title");
				}
				else
				{
					locString = new LocString("static_hover_tips", "NETWORK_PROBLEM_HOST.description");
					title = new LocString("static_hover_tips", "NETWORK_PROBLEM_HOST.title");
				}
				locString.Add("Player", PlatformUtil.GetPlayerName(RunManager.Instance.NetService.Platform, Player.NetId));
				NHoverTipSet.CreateAndShow(this, new HoverTip(title, locString)).GlobalPosition = base.GlobalPosition + Vector2.Down * 80f;
			}
		}
		else
		{
			_healthBar.FadeOutHpLabel(0.5f, 0f);
			_selectionReticle.OnDeselect();
			NHoverTipSet.Remove(this);
		}
	}

	private void BlockChanged(int oldBlock, int blockGain)
	{
		if (oldBlock == 0 && blockGain > 0)
		{
			_healthBar.AnimateInBlock(oldBlock, blockGain);
		}
		_healthBar.RefreshValues();
	}

	private void RefreshConnectedState(ulong _)
	{
		RefreshConnectedState();
	}

	private void RefreshConnectedState()
	{
		bool flag = RunManager.Instance.RunLobby.ConnectedPlayerIds.Contains(Player.NetId);
		_disconnectedIndicator.Visible = !flag;
		_characterIcon.SelfModulate = (flag ? Colors.White : StsColors.gray);
	}

	private void OnRelicObtained(RelicModel relic)
	{
		if (!LocalContext.IsMe(Player))
		{
			TaskHelper.RunSafely(AnimateRelicObtained(relic));
		}
	}

	private async Task AnimateRelicObtained(RelicModel relic)
	{
		await WaitUntilNextTweenTime();
		NRelic relicImage = NRelic.Create(relic, NRelic.IconSize.Small);
		relicImage.Model = relic;
		await ObtainedAnimation(relicImage);
		relicImage.QueueFreeSafely();
	}

	private void OnRelicRemoved(RelicModel relic)
	{
		if (!LocalContext.IsMe(Player))
		{
			TaskHelper.RunSafely(AnimateRelicRemoved(relic));
		}
	}

	private async Task AnimateRelicRemoved(RelicModel relic)
	{
		await WaitUntilNextTweenTime();
		NRelic relicImage = NRelic.Create(relic, NRelic.IconSize.Small);
		relicImage.Model = relic;
		await RemovedAnimation(relicImage);
		relicImage.QueueFreeSafely();
	}

	private void OnCardObtained(CardModel card)
	{
		if (!LocalContext.IsMe(Player))
		{
			TaskHelper.RunSafely(AnimateCardObtained(card));
		}
	}

	private async Task AnimateCardObtained(CardModel card)
	{
		await WaitUntilNextTweenTime();
		NDeckHistoryEntry cardNode = NDeckHistoryEntry.Create(card, 1);
		await ObtainedAnimation(cardNode);
		cardNode.QueueFreeSafely();
	}

	private void OnCardRemovedFromDeck(CardModel card)
	{
		if (!LocalContext.IsMe(Player))
		{
			TaskHelper.RunSafely(AnimateCardRemovedFromDeck(card));
		}
	}

	private async Task AnimateCardRemovedFromDeck(CardModel card)
	{
		await WaitUntilNextTweenTime();
		NDeckHistoryEntry cardNode = NDeckHistoryEntry.Create(card, 1);
		await RemovedAnimation(cardNode);
		cardNode.QueueFreeSafely();
	}

	private void OnPotionProcured(PotionModel potion)
	{
		if (!LocalContext.IsMe(Player))
		{
			TaskHelper.RunSafely(AnimatePotionObtained(potion));
		}
	}

	private async Task AnimatePotionObtained(PotionModel potion)
	{
		await WaitUntilNextTweenTime();
		NPotion node = NPotion.Create(potion);
		await ObtainedAnimation(node);
		node.QueueFreeSafely();
	}

	private void OnPotionDiscarded(PotionModel potion)
	{
		if (!LocalContext.IsMe(Player))
		{
			TaskHelper.RunSafely(AnimatePotionDiscarded(potion));
		}
	}

	private async Task AnimatePotionDiscarded(PotionModel potion)
	{
		await WaitUntilNextTweenTime();
		NPotion node = NPotion.Create(potion);
		await RemovedAnimation(node);
		node.QueueFreeSafely();
	}

	private void OnPlayerVoteChanged(Player player, MapVote? _, MapVote? __)
	{
		RefreshPlayerReadyIndicator(player);
	}

	private void OnPlayerVotesCleared()
	{
		RefreshPlayerReadyIndicator(Player);
	}

	private void RefreshPlayerReadyIndicator(Player player, bool _)
	{
		RefreshPlayerReadyIndicator(player);
	}

	private void RefreshPlayerReadyIndicator(Player player)
	{
		if (CombatManager.Instance.IsInProgress)
		{
			_turnEndIndicator.Visible = CombatManager.Instance.IsPlayerReadyToEndTurn(Player);
		}
		else
		{
			_turnEndIndicator.Visible = RunManager.Instance.MapSelectionSynchronizer.GetVote(Player).HasValue;
		}
		if (_turnEndIndicator.Visible && player == Player)
		{
			FlashEndTurn();
		}
	}

	private void OnPlayerEndTurnPing(ulong playerId)
	{
		if (Player.NetId == playerId)
		{
			FlashEndTurn();
		}
	}

	private void FlashEndTurn()
	{
		NUiFlashVfx nUiFlashVfx = NUiFlashVfx.Create(_turnEndIndicator.Texture, _turnEndIndicator.SelfModulate);
		_turnEndIndicator.AddChildSafely(nUiFlashVfx);
		nUiFlashVfx.SetDeferred(Control.PropertyName.Size, _turnEndIndicator.Size);
		nUiFlashVfx.Position = Vector2.Zero;
		TaskHelper.RunSafely(nUiFlashVfx.StartVfx());
	}

	private void OnTurnStarted(CombatState _)
	{
		_turnEndIndicator.Visible = CombatManager.Instance.IsPlayerReadyToEndTurn(Player);
	}

	private void SetNextTweenTime()
	{
		ulong ticksMsec = Time.GetTicksMsec();
		if (_nextTweenTime > ticksMsec)
		{
			_nextTweenTime += 500uL;
		}
		else
		{
			_nextTweenTime = ticksMsec + 500;
		}
	}

	private async Task WaitUntilNextTweenTime()
	{
		ulong nextTweenTime = _nextTweenTime;
		SetNextTweenTime();
		if (nextTweenTime >= Time.GetTicksMsec())
		{
			double timeSec = (double)(nextTweenTime - Time.GetTicksMsec()) / 1000.0;
			await ToSignal(GetTree().CreateTimer(timeSec), SceneTreeTimer.SignalName.Timeout);
		}
	}

	private async Task ObtainedAnimation(Control node)
	{
		this.AddChildSafely(node);
		node.Position = new Vector2(base.Size.X + 40f, 0f);
		node.Scale = Vector2.One * 1.1f;
		Tween tween = node.CreateTween();
		tween.TweenProperty(node, "scale", Vector2.One, 0.30000001192092896).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Bounce);
		tween.TweenProperty(node, "position:x", base.Size.X - node.Size.X, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
			.SetDelay(0.30000001192092896);
		tween.TweenProperty(node, "modulate:a", 0f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
			.SetDelay(0.30000001192092896);
		await ToSignal(tween, Tween.SignalName.Finished);
	}

	private async Task RemovedAnimation(Control node)
	{
		this.AddChildSafely(node);
		node.Position = new Vector2(base.Size.X - node.Size.X, 0f);
		Color modulate = node.Modulate;
		modulate.A = 0f;
		node.Modulate = modulate;
		Tween tween = node.CreateTween();
		tween.TweenProperty(node, "position:x", base.Size.X + 40f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		tween.TweenProperty(node, "modulate:a", 1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		tween.TweenProperty(node, "scale:y", 0f, 0.30000001192092896).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
			.SetDelay(0.5);
		await ToSignal(tween, Tween.SignalName.Finished);
	}

	private void OnPlayerScreenChanged(ulong playerId, NetScreenType _)
	{
		if (Player.NetId != playerId || LocalContext.IsMe(Player))
		{
			return;
		}
		Texture2D locationIcon = RunManager.Instance.InputSynchronizer.GetScreenType(playerId).GetLocationIcon();
		if (_currentLocationIcon != locationIcon)
		{
			_currentLocationIcon = locationIcon;
			if (locationIcon == null)
			{
				TweenLocationIconAway();
			}
			else
			{
				TweenLocationIconIn(locationIcon);
			}
		}
	}

	private void TweenLocationIconAway()
	{
		_locationIconTween?.Kill();
		_locationIconTween = _locationIcon.CreateTween();
		_locationIconTween.TweenProperty(_locationIcon, "scale", Vector2.Zero, 0.4).SetTrans(Tween.TransitionType.Back).SetEase(Tween.EaseType.In);
		_locationIconTween.TweenCallback(Callable.From(() => _locationContainer.Visible = false));
	}

	private void TweenLocationIconIn(Texture2D? texture)
	{
		_locationIconTween?.Kill();
		_locationIconTween = _locationIcon.CreateTween();
		if (!_locationContainer.Visible)
		{
			_locationContainer.Visible = true;
			_locationIcon.Texture = texture;
			_locationIconTween.TweenProperty(_locationIcon, "scale", Vector2.One, 0.4).SetTrans(Tween.TransitionType.Back).SetEase(Tween.EaseType.Out);
			return;
		}
		_locationIconTween.TweenProperty(_locationIcon, "scale", Vector2.One * 0.5f, 0.2).SetTrans(Tween.TransitionType.Back).SetEase(Tween.EaseType.In);
		_locationIconTween.TweenCallback(Callable.From(() => _locationIcon.Texture = texture));
		_locationIconTween.TweenProperty(_locationIcon, "scale", Vector2.One, 0.3).SetTrans(Tween.TransitionType.Back).SetEase(Tween.EaseType.Out);
	}

	protected void OnFocus(NButton _)
	{
		_isMouseOver = true;
		UpdateHighlightedState();
		if (NTargetManager.Instance.IsInSelection && NTargetManager.Instance.AllowedToTargetNode(this))
		{
			NTargetManager.Instance.OnNodeHovered(this);
			_focusedWhileTargeting = true;
		}
	}

	protected void OnUnfocus(NButton _)
	{
		_isMouseOver = false;
		UpdateHighlightedState();
		if (_focusedWhileTargeting)
		{
			NTargetManager.Instance.OnNodeUnhovered(this);
		}
		_focusedWhileTargeting = false;
	}

	protected void OnRelease(NButton _)
	{
		if (!NTargetManager.Instance.IsInSelection && NTargetManager.Instance.LastTargetingFinishedFrame != GetTree().GetFrame())
		{
			NMultiplayerPlayerExpandedState screen = NMultiplayerPlayerExpandedState.Create(Player);
			NCapstoneContainer.Instance.Open(screen);
		}
	}
}
