using System;
using Godot;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;

namespace MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;

public abstract partial class NSubmenu : Control, IScreenContext
{
	private NBackButton _backButton;

	protected NSubmenuStack _stack;

	protected Control? _lastFocusedControl;

	public Control? DefaultFocusedControl => _lastFocusedControl ?? InitialFocusedControl;

	protected abstract Control? InitialFocusedControl { get; }

	public override void _Ready()
	{
		if (GetType() != typeof(NSubmenu))
		{
			Log.Error($"{GetType()}");
			throw new InvalidOperationException("Don't call base._Ready()! Call ConnectSignals() instead.");
		}
		ConnectSignals();
	}

	protected virtual void ConnectSignals()
	{
		_backButton = GetNode<NBackButton>("BackButton");
		_backButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(delegate
		{
			_stack.Pop();
		}));
		_backButton.Disable();
		Connect(CanvasItem.SignalName.VisibilityChanged, Callable.From(OnScreenVisibilityChange));
	}

	public void HideBackButtonImmediately()
	{
		_backButton.Disable();
		_backButton.MoveToHidePosition();
	}

	public void SetStack(NSubmenuStack stack)
	{
		_stack = stack;
	}

	private void OnScreenVisibilityChange()
	{
		if (base.Visible)
		{
			_backButton.MoveToHidePosition();
			_backButton.Enable();
			OnSubmenuShown();
		}
		else
		{
			_lastFocusedControl = GetViewport()?.GuiGetFocusOwner();
			_backButton.Disable();
			OnSubmenuHidden();
		}
	}

	protected virtual void OnSubmenuShown()
	{
	}

	protected virtual void OnSubmenuHidden()
	{
	}

	public virtual void OnSubmenuOpened()
	{
	}

	public virtual void OnSubmenuClosed()
	{
		_lastFocusedControl = null;
	}
}
