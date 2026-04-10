using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Extensions;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Screens.Capstones;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Potions;

public partial class NPotionPopup : Control
{
	private NPotionHolder _holder;

	private Control _popupContainer;

	private NPotionPopupButton _useButton;

	private NPotionPopupButton _discardButton;

	private Control _hoverTipBounds;

	private Tween? _tween;

	private bool _markedForRemoval;

	private PotionModel? Potion => _holder.Potion?.Model;

	public bool IsUsable => _useButton.IsEnabled;

	private static string ScenePath => SceneHelper.GetScenePath("/potions/potion_popup");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	private bool InACardSelectScreen
	{
		get
		{
			NPlayerHand? instance = NPlayerHand.Instance;
			if (instance == null || !instance.IsInCardSelection)
			{
				return NOverlayStack.Instance?.Peek() is ICardSelector;
			}
			return true;
		}
	}

	public static NPotionPopup Create(NPotionHolder holder)
	{
		NPotionPopup nPotionPopup = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NPotionPopup>(PackedScene.GenEditState.Disabled);
		nPotionPopup._holder = holder;
		return nPotionPopup;
	}

	public override void _Ready()
	{
		base.GlobalPosition = _holder.GlobalPosition + Vector2.Down * _holder.Size.Y * 1.5f + Vector2.Right * _holder.Size * 0.5f + Vector2.Left * base.Size * 0.5f;
		_hoverTipBounds = GetNode<Control>("%HoverTipBounds");
		NHoverTipSet.CreateAndShow(_hoverTipBounds, _holder.Potion.Model.HoverTips, HoverTipAlignment.Right);
		NHoverTipSet.shouldBlockHoverTips = true;
		_popupContainer = GetNode<Control>("%Container");
		_useButton = GetNode<NPotionPopupButton>("%UseButton");
		_useButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnUseButtonPressed));
		_discardButton = GetNode<NPotionPopupButton>("%DiscardButton");
		_discardButton.SetLocKey("POTION_POPUP.discard");
		_discardButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnDiscardButtonPressed));
		_useButton.FocusNeighborLeft = _useButton.GetPath();
		_useButton.FocusNeighborRight = _useButton.GetPath();
		_useButton.FocusNeighborTop = _useButton.GetPath();
		_useButton.FocusNeighborBottom = _discardButton.GetPath();
		_discardButton.FocusNeighborLeft = _discardButton.GetPath();
		_discardButton.FocusNeighborRight = _discardButton.GetPath();
		_discardButton.FocusNeighborTop = _useButton.GetPath();
		_discardButton.FocusNeighborBottom = _discardButton.GetPath();
		if (Potion == null || Potion.IsQueued || Potion.Owner.Creature.IsDead)
		{
			_useButton.Disable();
			_discardButton.Disable();
		}
		else
		{
			switch (Potion.Usage)
			{
			case PotionUsage.None:
				throw new InvalidOperationException("No potions should have 'None' usage.");
			case PotionUsage.CombatOnly:
				CombatManager.Instance.StateTracker.CombatStateChanged += OnCombatStateChanged;
				CombatManager.Instance.TurnStarted += OnTurnStarted;
				CombatManager.Instance.PlayerEndedTurn += OnPlayerEndTurnStatusChanged;
				CombatManager.Instance.PlayerUnendedTurn += OnPlayerEndTurnStatusChanged;
				if (NOverlayStack.Instance != null)
				{
					NOverlayStack.Instance.Changed += Remove;
				}
				else
				{
					Log.Warn("NOverlayStack.Instance was null when creating potion popup");
				}
				if (NCapstoneContainer.Instance != null)
				{
					NCapstoneContainer.Instance.Changed += Remove;
				}
				else
				{
					Log.Warn("NCapstoneContainer.Instance was null when creating potion popup");
				}
				RefreshUseButton();
				break;
			case PotionUsage.AnyTime:
				_useButton.Enable();
				break;
			case PotionUsage.Automatic:
				_useButton.Disable();
				break;
			default:
				throw new ArgumentOutOfRangeException("Usage");
			}
			if (!Potion.Owner.CanRemovePotions)
			{
				_useButton.Disable();
				_discardButton.Disable();
			}
			if (!Potion.PassesCustomUsabilityCheck)
			{
				_useButton.Disable();
			}
			if (_useButton.IsEnabled)
			{
				_useButton.TryGrabFocus();
			}
			else if (_discardButton.IsEnabled)
			{
				_discardButton.TryGrabFocus();
			}
			else
			{
				this.TryGrabFocus();
			}
		}
		if (Potion == null)
		{
			_useButton.SetLocKey("POTION_POPUP.drink");
		}
		else
		{
			TargetType targetType = Potion.TargetType;
			bool flag = ((targetType == TargetType.AnyEnemy || targetType == TargetType.TargetedNoCreature) ? true : false);
			if (flag || Potion.CanThrowAtAlly())
			{
				_useButton.SetLocKey("POTION_POPUP.throw");
			}
			else
			{
				_useButton.SetLocKey("POTION_POPUP.drink");
			}
		}
		_tween?.Kill();
		base.Modulate = Colors.Transparent;
		_popupContainer.Position += Vector2.Up * 25f;
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "modulate", Colors.White, 0.10000000149011612).SetTrans(Tween.TransitionType.Sine);
		_tween.TweenProperty(_popupContainer, "position:y", _popupContainer.Position.Y + 25f, 0.15000000596046448).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Sine);
	}

	private void OnUseButtonPressed(NButton _)
	{
		TaskHelper.RunSafely(UsePotion());
		Remove();
	}

	private async Task UsePotion()
	{
		if (Potion == null)
		{
			return;
		}
		PotionModel potion = Potion;
		potion.BeforeUse += DisableHolder;
		try
		{
			await _holder.UsePotion();
		}
		finally
		{
			potion.BeforeUse -= DisableHolder;
		}
		void DisableHolder()
		{
			_holder.DisableUntilPotionRemoved();
		}
	}

	private void OnDiscardButtonPressed(NButton _)
	{
		if (Potion != null)
		{
			Player owner = Potion.Owner;
			int num = owner.PotionSlots.IndexOf<PotionModel>(_holder.Potion.Model);
			if (num < 0)
			{
				throw new InvalidOperationException($"Tried to discard potion {_holder.Potion.Model} but it's not in the player's belt!");
			}
			_holder.DisableUntilPotionRemoved();
			DiscardPotionGameAction action = new DiscardPotionGameAction(owner, (uint)num, CombatManager.Instance.IsInProgress);
			RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(action);
		}
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (inputEvent is InputEventMouseButton { ButtonIndex: var buttonIndex } inputEventMouseButton)
		{
			bool flag = (((ulong)(buttonIndex - 1) <= 1uL) ? true : false);
			if (flag && inputEventMouseButton.IsReleased())
			{
				Remove();
			}
		}
		else if (inputEvent.IsActionPressed(MegaInput.cancel))
		{
			Remove();
			_holder.TryGrabFocus();
		}
	}

	public override void _ExitTree()
	{
		if (!_markedForRemoval)
		{
			_markedForRemoval = true;
			NHoverTipSet.shouldBlockHoverTips = false;
			NHoverTipSet.Remove(_hoverTipBounds);
			DisconnectSignals();
		}
		_tween?.Kill();
	}

	public void Remove()
	{
		if (!_markedForRemoval)
		{
			_markedForRemoval = true;
			NHoverTipSet.shouldBlockHoverTips = false;
			NHoverTipSet.Remove(_hoverTipBounds);
			DisconnectSignals();
			Callable.From(delegate
			{
				_useButton.Disable();
				_discardButton.Disable();
			}).CallDeferred();
			_tween?.Kill();
			_tween = CreateTween().SetParallel();
			_tween.TweenProperty(this, "modulate", Colors.Transparent, 0.10000000149011612).SetTrans(Tween.TransitionType.Sine);
			_tween.TweenProperty(_popupContainer, "position:y", -25f, 0.20000000298023224).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Sine);
			_tween.Chain().TweenCallback(Callable.From(this.QueueFreeSafely));
		}
	}

	private void DisconnectSignals()
	{
		CombatManager.Instance.StateTracker.CombatStateChanged -= OnCombatStateChanged;
		CombatManager.Instance.TurnStarted -= OnTurnStarted;
		CombatManager.Instance.PlayerEndedTurn -= OnPlayerEndTurnStatusChanged;
		CombatManager.Instance.PlayerUnendedTurn -= OnPlayerEndTurnStatusChanged;
		if (NOverlayStack.Instance != null)
		{
			NOverlayStack.Instance.Changed -= Remove;
		}
		if (NCapstoneContainer.Instance != null)
		{
			NCapstoneContainer.Instance.Changed -= Remove;
		}
	}

	private void OnTurnStarted(CombatState _)
	{
		RefreshUseButton();
	}

	private void OnPlayerEndTurnStatusChanged(Player _, bool __)
	{
		RefreshUseButton();
	}

	private void OnPlayerEndTurnStatusChanged(Player _)
	{
		RefreshUseButton();
	}

	private void OnCombatStateChanged(CombatState _)
	{
		RefreshUseButton();
	}

	private void RefreshUseButton()
	{
		if (!_markedForRemoval)
		{
			Creature creature = Potion?.Owner.Creature;
			if (creature != null && CombatManager.Instance.IsInProgress && creature.CombatState?.CurrentSide == creature.Side && !InACardSelectScreen && !CombatManager.Instance.PlayerActionsDisabled)
			{
				_useButton.Enable();
			}
			else
			{
				_useButton.Disable();
			}
		}
	}
}
