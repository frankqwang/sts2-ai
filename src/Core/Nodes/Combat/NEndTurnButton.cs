using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Ftue;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NEndTurnButton : NButton
{
	private enum State
	{
		Enabled,
		Disabled,
		Hidden
	}

	private static readonly StringName _v = new StringName("v");

	private const float _flyInOutDuration = 0.5f;

	private static readonly LocString _endTurnLoc = new LocString("gameplay_ui", "END_TURN_BUTTON");

	private CombatState? _combatState;

	private CardPile? _playerHand;

	private State _state = State.Hidden;

	private bool _isShiny;

	private HoverTip _hoverTip;

	private Control _visuals;

	private Texture2D _glowTexture;

	private Texture2D _normalTexture;

	private TextureRect _image;

	private ShaderMaterial _hsv;

	private Control _glow;

	private Control _glowVfx;

	private MegaLabel _label;

	private NCombatUi _combatUi;

	private Viewport _viewport;

	private NMultiplayerVoteContainer _playerIconContainer;

	private NEndTurnLongPressBar _longPressBar;

	private float _pulseTimer = 1f;

	private static readonly Vector2 _hoverTipOffset = new Vector2(-76f, -302f);

	private static readonly Vector2 _showPosRatio = new Vector2(1604f, 846f) / NGame.devResolution;

	private static readonly Vector2 _hidePosRatio = _showPosRatio + new Vector2(0f, 250f) / NGame.devResolution;

	private Tween? _positionTween;

	private Tween? _hoverTween;

	private Tween? _glowVfxTween;

	private Tween? _glowEnableTween;

	private const int _ftueDisableEndTurnCount = 3;

	private int _endTurnWithNoPlayableCardsCount;

	private static string EndTurnButtonPath => "res://images/packed/combat_ui/end_turn_button.png";

	private static string EndTurnButtonGlowPath => "res://images/packed/combat_ui/end_turn_button_glow.png";

	public static IEnumerable<string> AssetPaths
	{
		get
		{
			List<string> list = new List<string>();
			list.Add(EndTurnButtonPath);
			list.Add(EndTurnButtonGlowPath);
			list.AddRange(NMultiplayerVoteContainer.AssetPaths);
			return new _003C_003Ez__ReadOnlyList<string>(list);
		}
	}

	private bool CanTurnBeEnded
	{
		get
		{
			if (!NCombatRoom.Instance.Ui.Hand.InCardPlay)
			{
				return NCombatRoom.Instance.Ui.Hand.CurrentMode == NPlayerHand.Mode.Play;
			}
			return false;
		}
	}

	private Vector2 ShowPos => _showPosRatio * _viewport.GetVisibleRect().Size;

	private Vector2 HidePos => _hidePosRatio * _viewport.GetVisibleRect().Size;

	protected override string[] Hotkeys => new string[1] { MegaInput.accept };

	public override void _Ready()
	{
		ConnectSignals();
		_visuals = GetNode<Control>("Visuals");
		_image = GetNode<TextureRect>("Visuals/Image");
		_hsv = (ShaderMaterial)_image.Material;
		_glow = GetNode<TextureRect>("Visuals/Glow");
		_glowVfx = GetNode<TextureRect>("Visuals/GlowVfx");
		_label = GetNode<MegaLabel>("Visuals/Label");
		_playerIconContainer = GetNode<NMultiplayerVoteContainer>("PlayerIconContainer");
		_longPressBar = GetNode<NEndTurnLongPressBar>("%Bar");
		_longPressBar.Init(this);
		_isEnabled = false;
		_combatUi = GetParent<NCombatUi>();
		_viewport = GetViewport();
		_hoverTip = new HoverTip(new LocString("static_hover_tips", "END_TURN.title"), new LocString("static_hover_tips", "END_TURN.description"));
		_glowTexture = PreloadManager.Cache.GetCompressedTexture2D(EndTurnButtonGlowPath);
		_normalTexture = PreloadManager.Cache.GetCompressedTexture2D(EndTurnButtonPath);
	}

	public override void _EnterTree()
	{
		base._EnterTree();
		CombatManager.Instance.TurnStarted += OnTurnStarted;
		CombatManager.Instance.AboutToSwitchToEnemyTurn += OnAboutToSwitchToEnemyTurn;
		CombatManager.Instance.PlayerEndedTurn += AfterPlayerEndedTurn;
		CombatManager.Instance.PlayerUnendedTurn += AfterPlayerUnendedTurn;
		CombatManager.Instance.StateTracker.CombatStateChanged += OnCombatStateChanged;
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		_positionTween?.Kill();
		_hoverTween?.Kill();
		CombatManager.Instance.TurnStarted -= OnTurnStarted;
		CombatManager.Instance.AboutToSwitchToEnemyTurn -= OnAboutToSwitchToEnemyTurn;
		CombatManager.Instance.PlayerEndedTurn -= AfterPlayerEndedTurn;
		CombatManager.Instance.PlayerUnendedTurn -= AfterPlayerUnendedTurn;
		CombatManager.Instance.StateTracker.CombatStateChanged -= OnCombatStateChanged;
	}

	public void Initialize(CombatState state)
	{
		_combatState = state;
		_playerHand = PileType.Hand.GetPile(LocalContext.GetMe(_combatState));
		_playerIconContainer.Initialize(ShouldDisplayPlayerIcon, _combatState.Players);
	}

	private bool ShouldDisplayPlayerIcon(Player player)
	{
		return CombatManager.Instance.IsPlayerReadyToEndTurn(player);
	}

	private bool PlayerCanTakeAction(Player player)
	{
		if (player.Creature.IsAlive)
		{
			if (CombatManager.Instance.PlayersTakingExtraTurn.Count != 0)
			{
				return CombatManager.Instance.PlayersTakingExtraTurn.Contains(player);
			}
			return true;
		}
		return false;
	}

	private void AfterPlayerEndedTurn(Player player, bool canBackOut)
	{
		_playerIconContainer.RefreshPlayerVotes();
		if (LocalContext.IsMe(player))
		{
			StartOrStopPulseVfx();
			Player me = LocalContext.GetMe(player.Creature.CombatState);
			if (!CombatManager.Instance.AllPlayersReadyToEndTurn() && PlayerCanTakeAction(me) && canBackOut)
			{
				SetState(State.Enabled);
				_label.SetTextAutoSize(new LocString("gameplay_ui", "UNDO_END_TURN_BUTTON").GetFormattedText());
			}
			else
			{
				SetState(State.Disabled);
			}
		}
		if (CombatManager.Instance.AllPlayersReadyToEndTurn())
		{
			SetState(State.Disabled);
		}
	}

	private void AfterPlayerUnendedTurn(Player player)
	{
		_playerIconContainer.RefreshPlayerVotes();
		if (LocalContext.IsMe(player) && PlayerCanTakeAction(player))
		{
			SetState(State.Enabled);
			_endTurnLoc.Add("turnNumber", player.Creature.CombatState.RoundNumber);
			_label.SetTextAutoSize(_endTurnLoc.GetFormattedText());
			StartOrStopPulseVfx();
		}
	}

	private void OnAboutToSwitchToEnemyTurn(CombatState _)
	{
		SetState(State.Hidden);
	}

	private void OnTurnStarted(CombatState state)
	{
		if (state.CurrentSide == CombatSide.Player && CombatManager.Instance.IsInProgress)
		{
			_playerIconContainer.RefreshPlayerVotes(animate: false);
			Player me = LocalContext.GetMe(state);
			_endTurnLoc.Add("turnNumber", state.RoundNumber);
			_label.SetTextAutoSize(_endTurnLoc.GetFormattedText());
			if (PlayerCanTakeAction(me))
			{
				SetState(State.Enabled);
				return;
			}
			AnimIn();
			SetState(State.Disabled);
		}
	}

	private void OnCombatStateChanged(CombatState combatState)
	{
		StartOrStopPulseVfx();
	}

	private void StartOrStopPulseVfx()
	{
		bool flag = !HasPlayableCard() && !CombatManager.Instance.IsPlayerReadyToEndTurn(LocalContext.GetMe(_combatState)) && _state == State.Enabled;
		if (_isShiny)
		{
			if (!flag)
			{
				_isShiny = false;
				_glowEnableTween?.Kill();
				_glowEnableTween = CreateTween().SetParallel();
				_glowEnableTween.TweenProperty(_glow, "modulate:a", 0f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
				_glowVfxTween?.Kill();
				_glowVfxTween = CreateTween();
				_glowVfxTween.TweenProperty(_glowVfx, "modulate:a", 0f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			}
		}
		else if (flag)
		{
			_isShiny = true;
			_glowVfxTween?.Kill();
			GlowPulse();
			_glowEnableTween?.Kill();
			_glowEnableTween = CreateTween().SetParallel();
			_glowEnableTween.TweenProperty(_glow, "modulate:a", 0.75f, 0.8).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back);
			_glowEnableTween.TweenProperty(_glow, "scale", Vector2.One * 0.5f, 0.8).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back)
				.From(Vector2.One * 0.45f);
		}
	}

	private void GlowPulse()
	{
		_glowVfxTween = CreateTween().SetParallel().SetLoops();
		_glowVfxTween.TweenProperty(_glowVfx, "scale", Vector2.One * 0.7f, 1.5).From(Vector2.One * 0.5f).SetEase(Tween.EaseType.Out)
			.SetTrans(Tween.TransitionType.Quart);
		_glowVfxTween.TweenProperty(_glowVfx, "modulate:a", 0f, 1.5).From(0.4f);
	}

	protected override void OnRelease()
	{
		if (!ShouldShowPlayableCardsFtue())
		{
			if (SaveManager.Instance.PrefsSave.IsLongPressEnabled)
			{
				_longPressBar.CancelPress();
			}
			else
			{
				CallReleaseLogic();
			}
		}
	}

	public void CallReleaseLogic()
	{
		if (CanTurnBeEnded)
		{
			_glowEnableTween?.Kill();
			_glowEnableTween = CreateTween().SetParallel();
			_glowEnableTween.TweenProperty(_glow, "modulate:a", 0f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			Player me = LocalContext.GetMe(_combatState);
			int roundNumber = me.Creature.CombatState.RoundNumber;
			if (!CombatManager.Instance.IsPlayerReadyToEndTurn(me))
			{
				SetState(State.Disabled);
				RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new EndPlayerTurnAction(me, roundNumber));
			}
			else
			{
				SetState(State.Disabled);
				RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new UndoEndPlayerTurnAction(me, roundNumber));
			}
		}
	}

	public void SecretEndTurnLogicViaFtue()
	{
		_glowEnableTween?.Kill();
		_glowEnableTween = CreateTween().SetParallel();
		_glowEnableTween.TweenProperty(_glow, "modulate:a", 0f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		Player me = LocalContext.GetMe(_combatState);
		RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new EndPlayerTurnAction(me, me.Creature.CombatState.RoundNumber));
	}

	private bool ShouldShowPlayableCardsFtue()
	{
		if (SaveManager.Instance.SeenFtue("can_play_cards_ftue"))
		{
			return false;
		}
		bool flag = LocalContext.GetMe(_combatState).PlayerCombatState.HasCardsToPlay();
		if (flag)
		{
			NModalContainer.Instance.Add(NCanPlayCardsFtue.Create());
			SaveManager.Instance.MarkFtueAsComplete("can_play_cards_ftue");
		}
		else
		{
			_endTurnWithNoPlayableCardsCount++;
			if (_endTurnWithNoPlayableCardsCount == 3)
			{
				Log.Info($"Ended {3} turns without cards left to play. Good job! Disabling can_play_cards ftue.");
				SaveManager.Instance.MarkFtueAsComplete("can_play_cards_ftue");
			}
		}
		return flag;
	}

	protected override void OnEnable()
	{
		base.OnEnable();
		_hoverTween?.CustomStep(999.0);
		_image.Texture = _normalTexture;
		_image.Modulate = Colors.White;
		_label.Modulate = StsColors.cream;
	}

	protected override void OnDisable()
	{
		base.OnDisable();
		NHoverTipSet.Remove(this);
		_hoverTween?.CustomStep(999.0);
		_image.Modulate = StsColors.gray;
		_label.Modulate = StsColors.gray;
		StartOrStopPulseVfx();
	}

	private void AnimOut()
	{
		_hoverTween?.Kill();
		_positionTween?.Kill();
		_positionTween = CreateTween();
		_positionTween.TweenProperty(this, "position", HidePos, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	private void AnimIn()
	{
		_positionTween?.Kill();
		_positionTween = CreateTween();
		_positionTween.TweenProperty(this, "position", ShowPos, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back);
	}

	public void OnCombatEnded()
	{
		SetState(State.Hidden);
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_hoverTween?.Kill();
		_hsv.SetShaderParameter(_v, 1.5);
		_visuals.Position = new Vector2(0f, -2f);
		Player me = LocalContext.GetMe(_combatState);
		if (!CombatManager.Instance.IsPlayerReadyToEndTurn(me))
		{
			_label.Modulate = (me.PlayerCombatState.HasCardsToPlay() ? StsColors.red : Colors.Cyan);
			_combatUi.Hand.FlashPlayableHolders();
			NHoverTipSet.CreateAndShow(this, _hoverTip).GlobalPosition = base.GlobalPosition + _hoverTipOffset;
		}
		else
		{
			_label.Modulate = StsColors.cream;
		}
	}

	private bool HasPlayableCard()
	{
		if (_playerHand == null)
		{
			return false;
		}
		foreach (CardModel card in _playerHand.Cards)
		{
			if (card.CanPlay())
			{
				return true;
			}
		}
		return false;
	}

	protected override void OnUnfocus()
	{
		if (SaveManager.Instance.PrefsSave.IsLongPressEnabled)
		{
			_longPressBar.CancelPress();
		}
		NHoverTipSet.Remove(this);
		_hoverTween?.Kill();
		_hoverTween = CreateTween().SetParallel();
		_hoverTween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_hoverTween.TweenProperty(_visuals, "position", Vector2.Zero, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_hoverTween.TweenProperty(_label, "modulate", base.IsEnabled ? StsColors.cream : StsColors.gray, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	protected override void OnPress()
	{
		if (CanTurnBeEnded)
		{
			if (SaveManager.Instance.PrefsSave.IsLongPressEnabled)
			{
				_longPressBar.StartPress();
			}
			_hoverTween?.Kill();
			_hoverTween = CreateTween().SetParallel();
			_hoverTween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			_hoverTween.TweenProperty(_visuals, "position", new Vector2(0f, 8f), 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
			_hoverTween.TweenProperty(_label, "modulate", Colors.DarkGray, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		}
	}

	private void UpdateShaderV(float value)
	{
		_hsv.SetShaderParameter(_v, value);
	}

	private void SetState(State newState)
	{
		if (_state != newState)
		{
			if (newState == State.Hidden)
			{
				AnimOut();
			}
			if (newState == State.Enabled && _state == State.Hidden)
			{
				AnimIn();
			}
			_state = newState;
			RefreshEnabled();
		}
	}

	public void RefreshEnabled()
	{
		bool flag = NCombatRoom.Instance == null || NCombatRoom.Instance.Mode != CombatRoomMode.ActiveCombat || !ActiveScreenContext.Instance.IsCurrent(NCombatRoom.Instance) || NCombatRoom.Instance.Ui.Hand.IsInCardSelection;
		if (_state == State.Enabled && !flag)
		{
			Enable();
		}
		else
		{
			Disable();
		}
	}
}
