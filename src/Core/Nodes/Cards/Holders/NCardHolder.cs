using System;
using Godot;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.UI;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;

namespace MegaCrit.Sts2.Core.Nodes.Cards.Holders;

public abstract partial class NCardHolder : Control
{
	[Signal]
	public delegate void PressedEventHandler(NCardHolder cardHolder);

	[Signal]
	public delegate void AltPressedEventHandler(NCardHolder cardHolder);

	public static readonly Vector2 smallScale = Vector2.One * 0.8f;

	protected NClickableControl _hitbox;

	protected bool _isHovered;

	protected bool _isFocused;

	protected Tween? _hoverTween;

	private InputEventMouseButton? _currentPressedAction;

	protected bool _isClickable = true;

	protected virtual Vector2 HoverScale => Vector2.One;

	public virtual Vector2 SmallScale => smallScale;

	public NClickableControl Hitbox => _hitbox;

	public NCard? CardNode { get; protected set; }

	public virtual CardModel? CardModel => CardNode?.Model;

	public virtual bool IsShowingUpgradedCard => CardModel?.IsUpgraded ?? false;

	protected bool CanBeFocused => _isHovered;

	public override void _Ready()
	{
		if (GetType() != typeof(NCardHolder))
		{
			Log.Error($"{GetType()}");
			throw new InvalidOperationException("Don't call base._Ready()! Call ConnectSignals() instead.");
		}
		ConnectSignals();
	}

	public void SetClickable(bool isClickable)
	{
		_isClickable = isClickable;
	}

	protected void ConnectSignals()
	{
		if (CardNode != null)
		{
			CardNode.Position = Vector2.Zero;
		}
		Connect(Control.SignalName.FocusEntered, Callable.From(OnFocus));
		Connect(Control.SignalName.FocusExited, Callable.From(OnUnfocus));
		_hitbox = GetNode<NClickableControl>("%Hitbox");
		_hitbox.Connect(NClickableControl.SignalName.Focused, Callable.From<NClickableControl>(delegate
		{
			OnFocus();
		}));
		_hitbox.Connect(NClickableControl.SignalName.Unfocused, Callable.From<NClickableControl>(delegate
		{
			OnUnfocus();
		}));
		_hitbox.Connect(NClickableControl.SignalName.MousePressed, Callable.From<InputEvent>(OnMousePressed));
		_hitbox.Connect(NClickableControl.SignalName.MouseReleased, Callable.From<InputEvent>(OnMouseReleased));
		Connect(Node.SignalName.ChildExitingTree, Callable.From<Node>(OnChildExitingTree));
	}

	public override void _GuiInput(InputEvent inputEvent)
	{
		base._GuiInput(inputEvent);
		if (_isClickable && CardNode != null)
		{
			if (inputEvent.IsActionPressed(MegaInput.select))
			{
				SfxCmd.Play("event:/sfx/ui/clicks/ui_click");
				EmitSignal(SignalName.Pressed, this);
			}
			else if (inputEvent.IsActionPressed(MegaInput.accept))
			{
				SfxCmd.Play("event:/sfx/ui/clicks/ui_click");
				EmitSignal(SignalName.AltPressed, this);
			}
		}
	}

	protected virtual void SetCard(NCard node)
	{
		if (CardNode != null)
		{
			throw new InvalidOperationException("Cannot set a card node on a holder that already has one.");
		}
		CardNode = node;
		if (CardNode.GetParent() == null)
		{
			this.AddChildSafely(node);
		}
		else
		{
			node.Reparent(this);
		}
	}

	public void ReassignToCard(CardModel cardModel, PileType pileType, Creature? target, ModelVisibility visibility)
	{
		CardNode.Visibility = visibility;
		CardNode.Model = cardModel;
		CardNode.SetPreviewTarget(target);
		CardNode.UpdateVisuals(pileType, CardPreviewMode.Normal);
		OnCardReassigned();
	}

	protected virtual void OnCardReassigned()
	{
	}

	protected virtual void OnMousePressed(InputEvent inputEvent)
	{
		if (_currentPressedAction == null && inputEvent is InputEventMouseButton inputEventMouseButton && _isClickable)
		{
			MouseButton buttonIndex = inputEventMouseButton.ButtonIndex;
			if (((ulong)(buttonIndex - 1) <= 1uL) ? true : false)
			{
				SfxCmd.Play("event:/sfx/ui/clicks/ui_click");
			}
			_currentPressedAction = inputEventMouseButton;
		}
	}

	protected virtual void OnMouseReleased(InputEvent inputEvent)
	{
		if (CardNode == null || !_isHovered || _currentPressedAction == null)
		{
			return;
		}
		if (inputEvent is InputEventMouseButton inputEventMouseButton && _isClickable)
		{
			if (inputEventMouseButton.ButtonIndex != _currentPressedAction.ButtonIndex)
			{
				return;
			}
			if (inputEventMouseButton.ButtonIndex == MouseButton.Left)
			{
				EmitSignal(SignalName.Pressed, this);
			}
			else
			{
				EmitSignal(SignalName.AltPressed, this);
			}
		}
		_currentPressedAction = null;
	}

	protected virtual void OnFocus()
	{
		_isHovered = true;
		RefreshFocusState();
	}

	protected virtual void CreateHoverTips()
	{
		if (CardNode != null)
		{
			NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, CardNode.Model.HoverTips);
			nHoverTipSet.SetAlignmentForCardHolder(this);
		}
	}

	protected void ClearHoverTips()
	{
		NHoverTipSet.Remove(this);
	}

	protected virtual void OnUnfocus()
	{
		_isHovered = false;
		_currentPressedAction = null;
		RefreshFocusState();
	}

	protected void RefreshFocusState()
	{
		if (_isFocused != CanBeFocused)
		{
			_isFocused = CanBeFocused;
			DoCardHoverEffects(_isFocused);
		}
	}

	protected virtual void DoCardHoverEffects(bool isHovered)
	{
		if (isHovered)
		{
			_hoverTween?.Kill();
			base.Scale = HoverScale;
			if (CardNode.Visibility == ModelVisibility.Visible)
			{
				CreateHoverTips();
			}
		}
		else if (!isHovered)
		{
			_hoverTween?.Kill();
			_hoverTween = CreateTween();
			_hoverTween.TweenProperty(this, "scale", SmallScale, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			ClearHoverTips();
		}
	}

	private void OnChildExitingTree(Node node)
	{
		if (node == CardNode && node.GetParent() != this)
		{
			ClearHoverTips();
			CardNode = null;
		}
	}

	public virtual void Clear()
	{
		if (CardNode != null)
		{
			if (CardNode.GetParent() == this)
			{
				this.RemoveChildSafely(CardNode);
			}
			CardNode = null;
		}
	}
}
