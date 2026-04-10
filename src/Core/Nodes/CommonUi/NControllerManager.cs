using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.ControllerInput.ControllerConfigs;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;

namespace MegaCrit.Sts2.Core.Nodes.CommonUi;

public partial class NControllerManager : Node
{
	[Signal]
	public delegate void ControllerDetectedEventHandler();

	[Signal]
	public delegate void MouseDetectedEventHandler();

	[Signal]
	public delegate void ControllerTypeChangedEventHandler();

	private IControllerInputStrategy? _inputStrategy;

	private static readonly Vector2 _offscreenPos = Vector2.One * -1000f;

	private Vector2 _lastMousePosition;

	private MegaLabel _label;

	private Tween? _notifyTween;

	public static NControllerManager? Instance
	{
		get
		{
			if (NGame.Instance == null)
			{
				return null;
			}
			return NGame.Instance.InputManager.ControllerManager;
		}
	}

	public bool ShouldAllowControllerRebinding => _inputStrategy?.ShouldAllowControllerRebinding ?? true;

	public bool IsUsingController { get; private set; }

	public Dictionary<StringName, StringName> GetDefaultControllerInputMap
	{
		get
		{
			if (_inputStrategy == null)
			{
				return new SteamControllerConfig().DefaultControllerInputMap;
			}
			return _inputStrategy.GetDefaultControllerInputMap;
		}
	}

	public ControllerMappingType ControllerMappingType
	{
		get
		{
			if (_inputStrategy == null)
			{
				return ControllerMappingType.Default;
			}
			return _inputStrategy.ControllerConfig.ControllerMappingType;
		}
	}

	public async Task Init()
	{
		ActiveScreenContext.Instance.Updated += OnScreenContextChanged;
		_label = GetNode<MegaLabel>("Label");
		_label.Modulate = Colors.Transparent;
		_inputStrategy = new SteamControllerInputStrategy();
		await _inputStrategy.Init();
	}

	public override void _ExitTree()
	{
		ActiveScreenContext.Instance.Updated -= OnScreenContextChanged;
	}

	public override void _Process(double delta)
	{
		_inputStrategy?.ProcessInput();
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (IsUsingController)
		{
			CheckForMouseInput(inputEvent);
		}
		else
		{
			CheckForControllerInput(inputEvent);
		}
	}

	public void OnControllerTypeChanged()
	{
		EmitSignalControllerTypeChanged();
	}

	private void CheckForMouseInput(InputEvent inputEvent)
	{
		bool flag = inputEvent is InputEventMouseButton;
		bool flag2 = inputEvent is InputEventMouseMotion { Velocity: var velocity } && velocity.LengthSquared() > 100f;
		Viewport viewport = GetViewport();
		if (flag || flag2)
		{
			IsUsingController = false;
			Input.WarpMouse(_lastMousePosition);
			viewport?.GuiReleaseFocus();
			EmitSignal(SignalName.MouseDetected);
			ControlModeChanged();
		}
	}

	private void CheckForControllerInput(InputEvent inputEvent)
	{
		if (Controller.AllControllerInputs.Any((StringName i) => inputEvent.IsActionPressed(i)))
		{
			IsUsingController = true;
			Viewport viewport = GetViewport();
			if (viewport != null)
			{
				Vector2I vector2I = DisplayServer.MouseGetPosition();
				Vector2I vector2I2 = DisplayServer.WindowGetPosition();
				_lastMousePosition = new Vector2(vector2I.X - vector2I2.X, vector2I.Y - vector2I2.Y);
				viewport.WarpMouse(_offscreenPos);
			}
			ActiveScreenContext.Instance.FocusOnDefaultControl();
			EmitSignal(SignalName.ControllerDetected);
			ControlModeChanged();
			viewport?.SetInputAsHandled();
		}
	}

	private void ControlModeChanged()
	{
		_notifyTween?.Kill();
		_notifyTween = CreateTween();
		_notifyTween.TweenProperty(_label, "modulate", Colors.White, 0.25);
		_notifyTween.TweenInterval(0.5);
		_notifyTween.TweenProperty(_label, "modulate", Colors.Transparent, 0.75);
		if (IsUsingController)
		{
			_label.SetTextAutoSize(new LocString("main_menu_ui", "CONTROLLER_DETECTED").GetFormattedText());
		}
		else
		{
			_label.SetTextAutoSize(new LocString("main_menu_ui", "MOUSE_DETECTED").GetFormattedText());
		}
	}

	private void OnScreenContextChanged()
	{
		if (IsUsingController)
		{
			Callable.From(delegate
			{
				ActiveScreenContext.Instance.FocusOnDefaultControl();
			}).CallDeferred();
		}
		else
		{
			Vector2I vector2I = DisplayServer.MouseGetPosition();
			Vector2I vector2I2 = DisplayServer.WindowGetPosition();
			Input.WarpMouse(new Vector2(vector2I.X - vector2I2.X, vector2I.Y - vector2I2.Y));
		}
	}

	public Texture2D? GetHotkeyIcon(string hotkey)
	{
		return _inputStrategy?.GetHotkeyIcon(hotkey);
	}
}
