using System;
using Godot;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.CommonUi;

namespace MegaCrit.Sts2.Core.Nodes.GodotExtensions;

public partial class NButton : NClickableControl
{
	protected TextureRect? _controllerHotkeyIcon;

	protected virtual string? ClickedSfx => "event:/sfx/ui/clicks/ui_click";

	protected virtual string HoveredSfx => "event:/sfx/ui/clicks/ui_hover";

	protected virtual string[] Hotkeys => Array.Empty<string>();

	protected virtual string? ControllerIconHotkey
	{
		get
		{
			if (Hotkeys.Length == 0)
			{
				return null;
			}
			return Hotkeys[0];
		}
	}

	private bool HasControllerHotkey => Hotkeys.Length != 0;

	public override void _Ready()
	{
		if (GetType() != typeof(NButton))
		{
			Log.Error($"{GetType()}");
			throw new InvalidOperationException("Don't call base._Ready()! Call ConnectSignals() instead.");
		}
		ConnectSignals();
	}

	protected override void ConnectSignals()
	{
		base.ConnectSignals();
		if (HasControllerHotkey)
		{
			RegisterHotkeys();
		}
		_controllerHotkeyIcon = GetNodeOrNull<TextureRect>("%ControllerIcon");
		UpdateControllerButton();
	}

	public override void _EnterTree()
	{
		if (NControllerManager.Instance != null)
		{
			NControllerManager.Instance.Connect(NControllerManager.SignalName.MouseDetected, Callable.From(UpdateControllerButton));
			NControllerManager.Instance.Connect(NControllerManager.SignalName.ControllerDetected, Callable.From(UpdateControllerButton));
		}
		if (NInputManager.Instance != null)
		{
			NInputManager.Instance.Connect(NInputManager.SignalName.InputRebound, Callable.From(UpdateControllerButton));
		}
	}

	public override void _Input(InputEvent inputEvent)
	{
		CheckMouseDragThreshold(inputEvent);
	}

	protected override void OnPress()
	{
		if (ClickedSfx != null)
		{
			SfxCmd.Play(ClickedSfx);
		}
	}

	protected override void OnFocus()
	{
		SfxCmd.Play(HoveredSfx);
	}

	protected override void OnEnable()
	{
		Callable.From(RegisterHotkeys).CallDeferred();
		UpdateControllerButton();
	}

	protected override void OnDisable()
	{
		UnregisterHotkeys();
		UpdateControllerButton();
	}

	protected void UpdateControllerButton()
	{
		if (_controllerHotkeyIcon == null)
		{
			return;
		}
		NControllerManager instance = NControllerManager.Instance;
		if (instance == null)
		{
			return;
		}
		_controllerHotkeyIcon.Visible = instance.IsUsingController && _isEnabled;
		if (ControllerIconHotkey != null)
		{
			Texture2D hotkeyIcon = NInputManager.Instance.GetHotkeyIcon(ControllerIconHotkey);
			if (hotkeyIcon != null)
			{
				_controllerHotkeyIcon.Texture = hotkeyIcon;
			}
		}
	}

	protected void RegisterHotkeys()
	{
		if (HasControllerHotkey && _isEnabled)
		{
			string[] hotkeys = Hotkeys;
			foreach (string hotkey in hotkeys)
			{
				NHotkeyManager.Instance.PushHotkeyPressedBinding(hotkey, base.OnPressHandler);
				NHotkeyManager.Instance.PushHotkeyReleasedBinding(hotkey, base.OnReleaseHandler);
			}
		}
	}

	protected void UnregisterHotkeys()
	{
		if (HasControllerHotkey)
		{
			string[] hotkeys = Hotkeys;
			foreach (string hotkey in hotkeys)
			{
				NHotkeyManager.Instance.RemoveHotkeyPressedBinding(hotkey, base.OnPressHandler);
				NHotkeyManager.Instance.RemoveHotkeyReleasedBinding(hotkey, base.OnReleaseHandler);
			}
		}
	}

	public override void _ExitTree()
	{
		if (NControllerManager.Instance != null)
		{
			NControllerManager.Instance.Disconnect(NControllerManager.SignalName.ControllerDetected, Callable.From(UpdateControllerButton));
			NControllerManager.Instance.Disconnect(NControllerManager.SignalName.MouseDetected, Callable.From(UpdateControllerButton));
		}
		if (NInputManager.Instance != null)
		{
			NInputManager.Instance.Disconnect(NInputManager.SignalName.InputRebound, Callable.From(UpdateControllerButton));
		}
		UnregisterHotkeys();
	}
}
