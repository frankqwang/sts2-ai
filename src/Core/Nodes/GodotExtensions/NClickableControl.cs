using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.ControllerInput;

namespace MegaCrit.Sts2.Core.Nodes.GodotExtensions;

public partial class NClickableControl : Control
{
	[Signal]
	public delegate void ReleasedEventHandler(NClickableControl button);

	[Signal]
	public delegate void FocusedEventHandler(NClickableControl button);

	[Signal]
	public delegate void UnfocusedEventHandler(NClickableControl button);

	[Signal]
	public delegate void MouseReleasedEventHandler(InputEvent inputEvent);

	[Signal]
	public delegate void MousePressedEventHandler(InputEvent inputEvent);

	[Export(PropertyHint.None, "")]
	protected float _ignoreDragThreshold = -1f;

	protected bool _isEnabled = true;

	private bool _isHovered;

	private bool _isControllerFocused;

	private bool _isControllerNavigable;

	private Vector2 _beginDragPosition;

	private bool _isPressed;

	private static readonly StyleBoxEmpty _blankFocusStyle = new StyleBoxEmpty();

	protected virtual bool AllowFocusWhileDisabled => false;

	protected bool IsFocused { get; private set; }

	public bool IsEnabled => _isEnabled;

	protected virtual void ConnectSignals()
	{
		Connect(Control.SignalName.FocusEntered, Callable.From(OnFocusHandler));
		Connect(Control.SignalName.FocusExited, Callable.From(OnUnFocusHandler));
		Connect(Control.SignalName.MouseEntered, Callable.From(OnHoverHandler));
		Connect(Control.SignalName.MouseExited, Callable.From(OnUnhoverHandler));
		Connect(SignalName.MousePressed, Callable.From<InputEvent>(HandleMousePress));
		Connect(SignalName.MouseReleased, Callable.From<InputEvent>(HandleMouseRelease));
		Connect(CanvasItem.SignalName.VisibilityChanged, Callable.From(OnVisibilityChanged));
		AddThemeStyleboxOverride(ThemeConstants.Control.focus, _blankFocusStyle);
		_isControllerNavigable = base.FocusMode == FocusModeEnum.All;
		if (HasFocus())
		{
			OnFocusHandler();
		}
	}

	private void OnVisibilityChanged()
	{
		if (!IsVisibleInTree())
		{
			OnUnFocusHandler();
		}
	}

	private void OnFocusHandler()
	{
		_isControllerFocused = true;
		RefreshFocus();
	}

	private void OnUnFocusHandler()
	{
		_isControllerFocused = false;
		RefreshFocus();
	}

	private void HandleMousePress(InputEvent inputEvent)
	{
		if (_isEnabled && IsVisibleInTree() && IsFocused && inputEvent is InputEventMouseButton inputEventMouseButton && inputEventMouseButton.ButtonIndex == MouseButton.Left)
		{
			_isControllerFocused = false;
			_beginDragPosition = inputEventMouseButton.GlobalPosition;
			OnPressHandler();
		}
	}

	private void HandleMouseRelease(InputEvent inputEvent)
	{
		if (_isEnabled && IsVisibleInTree() && IsFocused && inputEvent is InputEventMouseButton inputEventMouseButton && inputEventMouseButton.ButtonIndex == MouseButton.Left)
		{
			OnReleaseHandler();
		}
	}

	private void OnHoverHandler()
	{
		_isHovered = true;
		if (!GetTree().Paused || NGame.IsReleaseGame())
		{
			RefreshFocus();
		}
	}

	private void OnUnhoverHandler()
	{
		_isHovered = false;
		if (!GetTree().Paused || NGame.IsReleaseGame())
		{
			RefreshFocus();
		}
	}

	protected void OnPressHandler()
	{
		_isPressed = true;
		OnPress();
	}

	protected void OnReleaseHandler()
	{
		if (_isPressed)
		{
			_isPressed = false;
			OnRelease();
			EmitSignal(SignalName.Released, this);
		}
	}

	private void RefreshFocus()
	{
		bool flag = (_isEnabled || AllowFocusWhileDisabled) && IsVisibleInTree() && (_isHovered || _isControllerFocused);
		if (IsFocused != flag)
		{
			IsFocused = flag;
			if (IsFocused)
			{
				EmitSignal(SignalName.Focused, this);
				OnFocus();
			}
			else
			{
				EmitSignal(SignalName.Unfocused, this);
				OnUnfocus();
			}
		}
	}

	protected virtual void OnFocus()
	{
	}

	protected virtual void OnUnfocus()
	{
	}

	protected virtual void OnPress()
	{
	}

	protected virtual void OnRelease()
	{
	}

	public override void _GuiInput(InputEvent inputEvent)
	{
		if (inputEvent is InputEventMouseButton inputEventMouseButton && _isEnabled)
		{
			MouseButton buttonIndex = inputEventMouseButton.ButtonIndex;
			if (((ulong)(buttonIndex - 1) <= 1uL) ? true : false)
			{
				EmitSignal(inputEventMouseButton.IsPressed() ? SignalName.MousePressed : SignalName.MouseReleased, inputEvent);
			}
		}
		if (inputEvent.IsActionPressed(MegaInput.select))
		{
			OnPressHandler();
		}
		else if (inputEvent.IsActionReleased(MegaInput.select))
		{
			OnReleaseHandler();
		}
	}

	protected void CheckMouseDragThreshold(InputEvent inputEvent)
	{
		if (!(_ignoreDragThreshold <= 0f) && _isPressed && inputEvent is InputEventMouseMotion { GlobalPosition: var globalPosition } && globalPosition.DistanceTo(_beginDragPosition) >= _ignoreDragThreshold)
		{
			_isPressed = false;
		}
	}

	public void DebugPress()
	{
		EmitSignal(SignalName.MousePressed, new InputEventMouseButton
		{
			ButtonIndex = MouseButton.Left,
			Pressed = true
		});
	}

	public void DebugRelease()
	{
		EmitSignal(SignalName.MouseReleased, new InputEventMouseButton
		{
			ButtonIndex = MouseButton.Left,
			Pressed = false
		});
	}

	public void ForceClick()
	{
		OnRelease();
		EmitSignal(SignalName.Released, this);
	}

	public void SetEnabled(bool enabled)
	{
		if (enabled)
		{
			Enable();
		}
		else
		{
			Disable();
		}
	}

	public void Enable()
	{
		if (!_isEnabled)
		{
			_isEnabled = true;
			base.FocusMode = (FocusModeEnum)(_isControllerNavigable ? 2 : 0);
			OnEnable();
			RefreshFocus();
			Callable.From(delegate
			{
				SetProcessInput(enable: true);
			}).CallDeferred();
		}
	}

	public void Disable()
	{
		if (_isEnabled)
		{
			_isEnabled = false;
			_isPressed = false;
			base.FocusMode = FocusModeEnum.None;
			OnDisable();
			RefreshFocus();
			SetProcessInput(enable: false);
		}
	}

	protected virtual void OnEnable()
	{
	}

	protected virtual void OnDisable()
	{
	}
}
