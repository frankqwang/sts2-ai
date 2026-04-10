using System;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Shops;

public abstract partial class NMerchantSlot : Control
{
	[Signal]
	public delegate void HoveredEventHandler(NMerchantSlot slot);

	[Signal]
	public delegate void UnhoveredEventHandler(NMerchantSlot slot);

	private bool _isHovered;

	private static readonly Vector2 _hoverScale = Vector2.One * 0.8f;

	private static readonly Vector2 _smallScale = Vector2.One * 0.65f;

	protected NClickableControl _hitbox;

	protected MegaLabel _costLabel;

	private Tween? _hoverTween;

	private Tween? _purchaseFailedTween;

	private NMerchantInventory? _merchantRug;

	private bool _ignoreMouseRelease;

	private float? _originalVisualPosition;

	public NClickableControl Hitbox => _hitbox;

	public abstract MerchantEntry Entry { get; }

	protected abstract CanvasItem Visual { get; }

	protected Player? Player => _merchantRug?.Inventory?.Player;

	public void Initialize(NMerchantInventory rug)
	{
		_merchantRug = rug;
		Player.GoldChanged += UpdateVisual;
		Connect(SignalName.Hovered, Callable.From<NMerchantSlot>(OnMerchantHandHovered));
		Connect(SignalName.Unhovered, Callable.From<NMerchantSlot>(OnMerchantHandUnhovered));
	}

	public override void _Ready()
	{
		if (GetType() != typeof(NMerchantSlot))
		{
			Log.Error($"{GetType()}");
			throw new InvalidOperationException("Don't call base._Ready()! Call ConnectSignals() instead.");
		}
		ConnectSignals();
	}

	protected virtual void ConnectSignals()
	{
		_hitbox = GetNode<NClickableControl>("%Hitbox");
		Connect(Control.SignalName.FocusEntered, Callable.From(OnFocus));
		Connect(Control.SignalName.FocusExited, Callable.From(OnUnfocus));
		_hitbox.Connect(Control.SignalName.MouseEntered, Callable.From(OnFocus));
		_hitbox.Connect(Control.SignalName.MouseExited, Callable.From(OnUnfocus));
		_hitbox.Connect(NClickableControl.SignalName.MousePressed, Callable.From<InputEvent>(OnMousePressed));
		_hitbox.Connect(NClickableControl.SignalName.MouseReleased, Callable.From<InputEvent>(OnMouseReleased));
		_costLabel = GetNode<MegaLabel>("%CostLabel");
	}

	public override void _ExitTree()
	{
		Disconnect(SignalName.Unhovered, Callable.From<NMerchantSlot>(OnMerchantHandUnhovered));
		_hitbox.Disconnect(Control.SignalName.MouseExited, Callable.From(OnUnfocus));
		Disconnect(Control.SignalName.FocusExited, Callable.From(OnUnfocus));
		_hoverTween?.Kill();
		if (Player != null)
		{
			Player.GoldChanged -= UpdateVisual;
		}
	}

	public override void _GuiInput(InputEvent inputEvent)
	{
		if (inputEvent.IsActionReleased(MegaInput.select))
		{
			TaskHelper.RunSafely(OnReleased());
		}
		else if (inputEvent.IsActionReleased(MegaInput.accept))
		{
			OnPreview();
			GetViewport().SetInputAsHandled();
		}
	}

	private void OnMousePressed(InputEvent inputEvent)
	{
		_ignoreMouseRelease = false;
	}

	private void OnMouseReleased(InputEvent inputEvent)
	{
		if (_isHovered && !_ignoreMouseRelease && inputEvent is InputEventMouseButton inputEventMouseButton)
		{
			if (inputEventMouseButton.ButtonIndex == MouseButton.Left)
			{
				TaskHelper.RunSafely(OnReleased());
			}
			else
			{
				OnPreview();
			}
		}
	}

	private void OnFocus()
	{
		_isHovered = true;
		_hoverTween?.Kill();
		base.Scale = _hoverScale;
		CreateHoverTip();
		EmitSignal(SignalName.Hovered, this);
	}

	private void OnUnfocus()
	{
		_isHovered = false;
		_hoverTween?.Kill();
		_hoverTween = CreateTween();
		_hoverTween.TweenProperty(this, "scale", _smallScale, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		ClearHoverTip();
		EmitSignal(SignalName.Unhovered, this);
	}

	private async Task OnReleased()
	{
		ClearHoverTip();
		await OnTryPurchase(_merchantRug?.Inventory);
		MerchantEntry entry = Entry;
		if (entry != null && entry.IsStocked && _isHovered)
		{
			CreateHoverTip();
		}
	}

	protected abstract Task OnTryPurchase(MerchantInventory? inventory);

	protected void TriggerMerchantHandToPointHere()
	{
		_merchantRug?.MerchantHand.PointAtTarget(base.GlobalPosition);
		_merchantRug?.MerchantHand.StopPointing(2f);
	}

	protected virtual void OnPreview()
	{
	}

	protected abstract void CreateHoverTip();

	protected void ClearHoverTip()
	{
		NHoverTipSet.Remove(this);
	}

	private void OnMerchantHandHovered(NMerchantSlot _)
	{
		_merchantRug?.MerchantHand.PointAtTarget(base.GlobalPosition);
	}

	private void OnMerchantHandUnhovered(NMerchantSlot _)
	{
		_merchantRug?.MerchantHand.StopPointing(2f);
	}

	protected void OnPurchaseFailed(PurchaseStatus status)
	{
		if (status == PurchaseStatus.Success)
		{
			return;
		}
		if (!_originalVisualPosition.HasValue)
		{
			if (Visual is Node2D node2D)
			{
				_originalVisualPosition = node2D.Position.X;
			}
			else if (Visual is Control control)
			{
				_originalVisualPosition = control.Position.X;
			}
		}
		_purchaseFailedTween?.Kill();
		_purchaseFailedTween = CreateTween();
		_purchaseFailedTween.TweenMethod(Callable.From<float>(WiggleAnimation), 0f, 2f, 0.4000000059604645).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Quad);
		SfxCmd.Play("event:/sfx/npcs/merchant/merchant_dissapointment");
	}

	protected virtual void UpdateVisual()
	{
		if (Entry.IsStocked)
		{
			_costLabel.SetTextAutoSize(Entry.Cost.ToString());
		}
	}

	private void WiggleAnimation(float progress)
	{
		if (Visual is Node2D node2D)
		{
			Vector2 position = node2D.Position;
			position.X = _originalVisualPosition.Value + (float)Math.Sin(progress * (float)Math.PI * 2f) * 10f;
			node2D.Position = position;
		}
		else if (Visual is Control control)
		{
			Vector2 position = control.Position;
			position.X = _originalVisualPosition.Value + (float)Math.Sin(progress * (float)Math.PI * 2f) * 10f;
			control.Position = position;
		}
	}
}
