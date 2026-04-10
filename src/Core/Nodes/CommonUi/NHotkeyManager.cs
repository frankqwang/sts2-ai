using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Nodes.Debug;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.CommonUi;

public partial class NHotkeyManager : Node
{
	private readonly Dictionary<StringName, List<Action>> _hotkeyPressedBindings = new Dictionary<StringName, List<Action>>();

	private readonly Dictionary<StringName, List<Action>> _hotkeyReleasedBindings = new Dictionary<StringName, List<Action>>();

	private Dictionary<Node, Action> _blockingScreens = new Dictionary<Node, Action>();

	public static NHotkeyManager? Instance
	{
		get
		{
			if (NGame.Instance == null)
			{
				return null;
			}
			return NGame.Instance.HotkeyManager;
		}
	}

	public void PushHotkeyPressedBinding(string hotkey, Action action)
	{
		if (!_hotkeyPressedBindings.ContainsKey(hotkey))
		{
			_hotkeyPressedBindings.Add(hotkey, new List<Action>());
		}
		if (!_hotkeyPressedBindings[hotkey].Contains(action))
		{
			_hotkeyPressedBindings[hotkey].Add(action);
		}
	}

	public void RemoveHotkeyPressedBinding(string hotkey, Action action)
	{
		if (_hotkeyPressedBindings.TryGetValue(hotkey, out List<Action> value))
		{
			value.Remove(action);
			if (_hotkeyPressedBindings[hotkey].Count == 0)
			{
				_hotkeyPressedBindings.Remove(hotkey);
			}
		}
	}

	public void PushHotkeyReleasedBinding(string hotkey, Action action)
	{
		if (!_hotkeyReleasedBindings.ContainsKey(hotkey))
		{
			_hotkeyReleasedBindings.Add(hotkey, new List<Action>());
		}
		if (!_hotkeyReleasedBindings[hotkey].Contains(action))
		{
			_hotkeyReleasedBindings[hotkey].Add(action);
		}
	}

	public void RemoveHotkeyReleasedBinding(string hotkey, Action action)
	{
		if (_hotkeyReleasedBindings.TryGetValue(hotkey, out List<Action> value))
		{
			value.Remove(action);
			if (_hotkeyReleasedBindings[hotkey].Count == 0)
			{
				_hotkeyReleasedBindings.Remove(hotkey);
			}
		}
	}

	public void AddBlockingScreen(Node screen)
	{
		Action action = delegate
		{
		};
		string[] allInputs = MegaInput.AllInputs;
		foreach (string hotkey in allInputs)
		{
			PushHotkeyPressedBinding(hotkey, action);
		}
		string[] allInputs2 = MegaInput.AllInputs;
		foreach (string hotkey2 in allInputs2)
		{
			PushHotkeyReleasedBinding(hotkey2, action);
		}
		_blockingScreens.Add(screen, action);
	}

	public void RemoveBlockingScreen(Node screen)
	{
		if (_blockingScreens.TryGetValue(screen, out Action value))
		{
			string[] allInputs = MegaInput.AllInputs;
			foreach (string hotkey in allInputs)
			{
				RemoveHotkeyPressedBinding(hotkey, value);
			}
			string[] allInputs2 = MegaInput.AllInputs;
			foreach (string hotkey2 in allInputs2)
			{
				RemoveHotkeyReleasedBinding(hotkey2, value);
			}
			_blockingScreens.Remove(screen);
		}
	}

	public override void _UnhandledInput(InputEvent inputEvent)
	{
		if (NDevConsole.Instance.Visible)
		{
			return;
		}
		Control control = GetViewport()?.GuiGetFocusOwner();
		if (control != null && ((control is LineEdit lineEdit && lineEdit.IsEditing()) || (control is NMegaTextEdit nMegaTextEdit && nMegaTextEdit.IsEditing())))
		{
			return;
		}
		foreach (KeyValuePair<StringName, List<Action>> hotkeyPressedBinding in _hotkeyPressedBindings)
		{
			if (inputEvent.IsActionPressed(hotkeyPressedBinding.Key) && !inputEvent.IsEcho())
			{
				Action action = hotkeyPressedBinding.Value.LastOrDefault();
				if (action != null)
				{
					Callable.From(action.Invoke).CallDeferred();
				}
			}
		}
		foreach (KeyValuePair<StringName, List<Action>> hotkeyReleasedBinding in _hotkeyReleasedBindings)
		{
			if (inputEvent.IsActionReleased(hotkeyReleasedBinding.Key))
			{
				Action action2 = hotkeyReleasedBinding.Value.LastOrDefault();
				if (action2 != null)
				{
					Callable.From(action2.Invoke).CallDeferred();
				}
			}
		}
	}
}
