using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Debug;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Nodes.Vfx;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NCombatUi : Control
{
	private NStarCounter _starCounter;

	private NEnergyCounter _energyCounter;

	private NCombatPilesContainer _combatPilesContainer;

	private readonly Dictionary<NCard, Vector2> _originalPlayContainerCardPositions = new Dictionary<NCard, Vector2>();

	private readonly Dictionary<NCard, Vector2> _originalPlayContainerCardScales = new Dictionary<NCard, Vector2>();

	private Tween? _playContainerPeekModeTween;

	private int _originalHandChildIndex;

	private CombatState _state;

	private static bool _isDebugSlowRewards;

	private static bool _isDebugHidden;

	private static bool _isDebugHidingHand;

	public Control EnergyCounterContainer { get; private set; }

	public NEndTurnButton EndTurnButton { get; private set; }

	private NPingButton PingButton { get; set; }

	public NDrawPileButton DrawPile => _combatPilesContainer.DrawPile;

	public NDiscardPileButton DiscardPile => _combatPilesContainer.DiscardPile;

	public NExhaustPileButton ExhaustPile => _combatPilesContainer.ExhaustPile;

	public NPlayerHand Hand { get; private set; }

	public Control PlayContainer { get; private set; }

	public NCardPlayQueue PlayQueue { get; private set; }

	public Control CardPreviewContainer { get; private set; }

	public NMessyCardPreviewContainer MessyCardPreviewContainer { get; private set; }

	private IEnumerable<NCard> PlayContainerCards => PlayContainer.GetChildren().OfType<NCard>();

	public static bool IsDebugHidingIntent { get; private set; }

	public static bool IsDebugHidingPlayContainer { get; private set; }

	public static bool IsDebugHidingHpBar { get; private set; }

	public static bool IsDebugHideTextVfx { get; private set; }

	public static bool IsDebugHideTargetingUi { get; private set; }

	public static bool IsDebugHideMpTargetingUi { get; private set; }

	public static bool IsDebugHideMpIntents { get; private set; }

	public event Action? DebugToggleIntent;

	public event Action? DebugToggleHpBar;

	public override void _Ready()
	{
		EnergyCounterContainer = GetNode<Control>("%EnergyCounterContainer");
		_starCounter = GetNode<NStarCounter>("%StarCounter");
		EndTurnButton = GetNode<NEndTurnButton>("%EndTurnButton");
		PingButton = GetNode<NPingButton>("%PingButton");
		_combatPilesContainer = GetNode<NCombatPilesContainer>("%CombatPileContainer");
		Hand = GetNode<NPlayerHand>("%Hand");
		PlayContainer = GetNode<Control>("%PlayContainer");
		PlayQueue = GetNode<NCardPlayQueue>("%PlayQueue");
		CardPreviewContainer = GetNode<Control>("%CardPreviewContainer");
		MessyCardPreviewContainer = GetNode<NMessyCardPreviewContainer>("%MessyCardPreviewContainer");
		if (!_isDebugHidden)
		{
			return;
		}
		foreach (Control item in GetChildren().OfType<Control>())
		{
			if (item != Hand)
			{
				item.Modulate = (_isDebugHidden ? Colors.Transparent : Colors.White);
			}
		}
	}

	public override void _ExitTree()
	{
		DisconnectSignals();
	}

	public void Activate(CombatState state)
	{
		CombatManager.Instance.CombatEnded += AnimOut;
		CombatManager.Instance.CombatEnded += PostCombatCleanUp;
		CombatManager.Instance.CombatWon += OnCombatWon;
		_state = state;
		Player me = LocalContext.GetMe(_state);
		_combatPilesContainer.Initialize(me);
		_starCounter.Initialize(me);
		EndTurnButton.Initialize(state);
		if (me.Character.ShouldAlwaysShowStarCounter)
		{
			EnergyCounterContainer.SetPosition(new Vector2(100f, 806f), keepOffsets: true);
		}
		_energyCounter = NEnergyCounter.Create(me);
		EnergyCounterContainer.AddChildSafely(_energyCounter);
		_starCounter.Reparent(_energyCounter);
		base.Visible = true;
		AnimIn();
	}

	public void Deactivate()
	{
		DisconnectSignals();
		base.Visible = false;
	}

	private void DisconnectSignals()
	{
		CombatManager.Instance.CombatEnded -= AnimOut;
		CombatManager.Instance.CombatEnded -= PostCombatCleanUp;
		CombatManager.Instance.CombatWon -= OnCombatWon;
	}

	public void AddToPlayContainer(NCard card)
	{
		card.GetParent()?.RemoveChildSafely(card);
		PlayContainer.AddChildSafely(card);
	}

	public NCard? GetCardFromPlayContainer(CardModel model)
	{
		return PlayContainerCards.FirstOrDefault((NCard n) => n.Model == model);
	}

	private void OnCombatWon(CombatRoom room)
	{
		if (room.Encounter.ShouldGiveRewards)
		{
			TaskHelper.RunSafely(ShowRewards(room));
		}
		else
		{
			TaskHelper.RunSafely(ProceedWithoutRewards());
		}
	}

	private async Task ProceedWithoutRewards()
	{
		await Cmd.Wait(1f);
		await RunManager.Instance.ProceedFromTerminalRewardsScreen();
	}

	private async Task ShowRewards(CombatRoom room)
	{
		float num = 0f;
		foreach (NCreature removingCreatureNode in NCombatRoom.Instance.RemovingCreatureNodes)
		{
			if (removingCreatureNode != null && removingCreatureNode.HasSpineAnimation && removingCreatureNode.IsPlayingDeathAnimation)
			{
				num = Math.Max(num, removingCreatureNode.GetCurrentAnimationTimeRemaining());
				continue;
			}
			MonsterModel monster = removingCreatureNode.Entity.Monster;
			if (monster != null && monster.HasDeathAnimLengthOverride)
			{
				num = Math.Max(num, removingCreatureNode.Entity.Monster.DeathAnimLengthOverride);
			}
		}
		if (_isDebugSlowRewards)
		{
			await Cmd.Wait(num + 3f);
		}
		else if (room.RoomType == RoomType.Boss)
		{
			await Cmd.CustomScaledWait(num * 0.5f, num + 1f);
		}
		else
		{
			await Cmd.CustomScaledWait(0.5f, num + 1f);
		}
		Player me = LocalContext.GetMe(_state);
		await RewardsCmd.OfferForRoomEnd(me, room);
	}

	private void AnimIn()
	{
		Hand.AnimIn();
		_energyCounter.AnimIn();
		_combatPilesContainer.AnimIn();
	}

	public void AnimOut(CombatRoom _)
	{
		Hand.AnimOut();
		PlayQueue.AnimOut();
		EndTurnButton.OnCombatEnded();
		PingButton.OnCombatEnded();
		_energyCounter.AnimOut();
		_combatPilesContainer.AnimOut();
	}

	private void PostCombatCleanUp(CombatRoom _)
	{
		Tween tween = CreateTween();
		tween.TweenProperty(PlayContainer, "modulate", Colors.Transparent, 0.25);
	}

	public void OnHandSelectModeEntered()
	{
		_originalHandChildIndex = Hand.GetIndex();
		Hand.MoveToFront();
		ActiveScreenContext.Instance.Update();
	}

	public void OnHandSelectModeExited()
	{
		MoveChild(Hand, _originalHandChildIndex);
		ActiveScreenContext.Instance.Update();
	}

	public void OnPeekButtonReady(NPeekButton peekButton)
	{
		peekButton.Connect(NPeekButton.SignalName.Toggled, Callable.From<NPeekButton>(OnPeekButtonToggled));
	}

	private void OnPeekButtonToggled(NPeekButton peekButton)
	{
		if (_playContainerPeekModeTween != null)
		{
			_playContainerPeekModeTween.Pause();
			_playContainerPeekModeTween.CustomStep(0.25);
			_playContainerPeekModeTween.Kill();
			_playContainerPeekModeTween = null;
		}
		Vector2 size = GetViewportRect().Size;
		if (peekButton.IsPeeking)
		{
			PlayQueue.Hide();
			foreach (NCard playContainerCard in PlayContainerCards)
			{
				_originalPlayContainerCardPositions[playContainerCard] = playContainerCard.Position;
				_originalPlayContainerCardScales[playContainerCard] = playContainerCard.Scale;
				Vector2 globalPosition = peekButton.CurrentCardMarker.GlobalPosition;
				Vector2 vector = playContainerCard.Scale * 0.5f;
				if (_playContainerPeekModeTween == null)
				{
					_playContainerPeekModeTween = CreateTween();
				}
				_playContainerPeekModeTween.TweenProperty(playContainerCard, "global_position", globalPosition, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
				_playContainerPeekModeTween.Parallel().TweenProperty(playContainerCard, "scale", vector, 0.25).SetEase(Tween.EaseType.Out)
					.SetTrans(Tween.TransitionType.Cubic);
			}
		}
		else
		{
			PlayQueue.Show();
			foreach (NCard playContainerCard2 in PlayContainerCards)
			{
				Vector2 value;
				Vector2 vector2 = (_originalPlayContainerCardPositions.TryGetValue(playContainerCard2, out value) ? value : (size * 0.5f));
				Vector2 value2;
				Vector2 vector3 = (_originalPlayContainerCardScales.TryGetValue(playContainerCard2, out value2) ? value2 : Vector2.One);
				if (_playContainerPeekModeTween == null)
				{
					_playContainerPeekModeTween = CreateTween();
				}
				_playContainerPeekModeTween.TweenProperty(playContainerCard2, "position", vector2, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
				_playContainerPeekModeTween.Parallel().TweenProperty(playContainerCard2, "scale", vector3, 0.25).SetEase(Tween.EaseType.Out)
					.SetTrans(Tween.TransitionType.Cubic);
			}
			_originalPlayContainerCardPositions.Clear();
			_originalPlayContainerCardScales.Clear();
		}
		ActiveScreenContext.Instance.Update();
	}

	public void Enable()
	{
		NPlayerHand hand = Hand;
		if (hand != null && hand.IsInCardSelection)
		{
			NPeekButton peekButton = hand.PeekButton;
			if (peekButton != null && !peekButton.IsPeeking)
			{
				_combatPilesContainer.Disable();
				goto IL_003c;
			}
		}
		_combatPilesContainer.Enable();
		goto IL_003c;
		IL_003c:
		if (_state.CurrentSide == CombatSide.Player)
		{
			EndTurnButton.RefreshEnabled();
		}
		if (_state.PlayerCreatures.Count > 1)
		{
			PingButton.RefreshEnabled();
		}
		NPlayerHand.Mode currentMode = Hand.CurrentMode;
		if ((uint)(currentMode - 2) <= 1u)
		{
			Hand.PeekButton.Enable();
		}
	}

	public void Disable()
	{
		_combatPilesContainer.Disable();
		EndTurnButton.RefreshEnabled();
		PingButton.RefreshEnabled();
		Hand.PeekButton.Disable();
		Hand.CancelAllCardPlay();
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (inputEvent.IsActionReleased(DebugHotkey.hideIntents))
		{
			IsDebugHidingIntent = !IsDebugHidingIntent;
			NGame.Instance.AddChildSafely(NFullscreenTextVfx.Create(IsDebugHidingIntent ? "Hide Intents" : "Show Intents"));
			this.DebugToggleIntent?.Invoke();
		}
		if (inputEvent.IsActionReleased(DebugHotkey.hideCombatUi))
		{
			_isDebugHidden = !_isDebugHidden;
			NGame.Instance.AddChildSafely(NFullscreenTextVfx.Create(_isDebugHidden ? "Hide Combat UI" : "Show Combat UI"));
			DebugHideCombatUi();
		}
		else if (inputEvent.IsActionReleased(DebugHotkey.hidePlayContainer))
		{
			IsDebugHidingPlayContainer = !IsDebugHidingPlayContainer;
			NGame.Instance.AddChildSafely(NFullscreenTextVfx.Create(IsDebugHidingPlayContainer ? "Hide Played Card" : "Show Played Card"));
			DebugHideCombatUi();
		}
		else if (inputEvent.IsActionReleased(DebugHotkey.hideHand))
		{
			_isDebugHidingHand = !_isDebugHidingHand;
			NGame.Instance.AddChildSafely(NFullscreenTextVfx.Create(_isDebugHidingHand ? "Hide Hand Cards" : "Show Hand Cards"));
			DebugHideCombatUi();
		}
		else if (inputEvent.IsActionReleased(DebugHotkey.hideHpBars))
		{
			IsDebugHidingHpBar = !IsDebugHidingHpBar;
			NGame.Instance.AddChildSafely(NFullscreenTextVfx.Create(IsDebugHidingHpBar ? "Hide HP Bars" : "Show HP Bars"));
			this.DebugToggleHpBar?.Invoke();
		}
		else if (inputEvent.IsActionReleased(DebugHotkey.hideTextVfx))
		{
			IsDebugHideTextVfx = !IsDebugHideTextVfx;
			NGame.Instance.AddChildSafely(NFullscreenTextVfx.Create(IsDebugHideTextVfx ? "Hide Text Vfx" : "Show Text Vfx"));
		}
		else if (inputEvent.IsActionReleased(DebugHotkey.hideTargetingUi))
		{
			IsDebugHideTargetingUi = !IsDebugHideTargetingUi;
			NGame.Instance.AddChildSafely(NFullscreenTextVfx.Create(IsDebugHideTargetingUi ? "Hide Targeting UI" : "Show Targeting UI"));
		}
		else if (inputEvent.IsActionReleased(DebugHotkey.slowRewards))
		{
			_isDebugSlowRewards = !_isDebugSlowRewards;
			NGame.Instance.AddChildSafely(NFullscreenTextVfx.Create(_isDebugSlowRewards ? "Slow Rewards Screens" : "Normal Rewards Screen"));
		}
		else if (inputEvent.IsActionReleased(DebugHotkey.hideMpTargeting))
		{
			IsDebugHideMpTargetingUi = !IsDebugHideMpTargetingUi;
			NGame.Instance.AddChildSafely(NFullscreenTextVfx.Create(IsDebugHideMpTargetingUi ? "Hide MP Targeting" : "Show MP Targeting"));
		}
		else if (inputEvent.IsActionReleased(DebugHotkey.hideMpIntents))
		{
			IsDebugHideMpIntents = !IsDebugHideMpIntents;
			NGame.Instance.AddChildSafely(NFullscreenTextVfx.Create(IsDebugHideMpIntents ? "Hide MP Intents" : "Show MP Intents"));
		}
	}

	private void DebugHideCombatUi()
	{
		foreach (Control item in GetChildren().OfType<Control>())
		{
			if (item == Hand)
			{
				item.Modulate = (_isDebugHidingHand ? Colors.Transparent : Colors.White);
			}
			else if (item.Name == PropertyName.PlayContainer)
			{
				item.Modulate = (IsDebugHidingPlayContainer ? Colors.Transparent : Colors.White);
			}
			else
			{
				item.Modulate = (_isDebugHidden ? Colors.Transparent : Colors.White);
			}
		}
	}
}
