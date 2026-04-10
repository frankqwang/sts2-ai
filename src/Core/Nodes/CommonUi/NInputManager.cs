using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Debug;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Debug;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.CommonUi;

public partial class NInputManager : Node
{
	[Signal]
	public delegate void InputReboundEventHandler();

	private readonly Dictionary<Key, StringName> _debugInputMap = new Dictionary<Key, StringName>
	{
		{
			Key.Key1,
			DebugHotkey.hideTopBar
		},
		{
			Key.Key2,
			DebugHotkey.hideIntents
		},
		{
			Key.Key3,
			DebugHotkey.hideCombatUi
		},
		{
			Key.Key4,
			DebugHotkey.hidePlayContainer
		},
		{
			Key.Key5,
			DebugHotkey.hideHand
		},
		{
			Key.Key6,
			DebugHotkey.hideHpBars
		},
		{
			Key.Key7,
			DebugHotkey.hideTextVfx
		},
		{
			Key.Key8,
			DebugHotkey.hideTargetingUi
		},
		{
			Key.Key9,
			DebugHotkey.slowRewards
		},
		{
			Key.Key0,
			DebugHotkey.hideVersionInfo
		},
		{
			Key.Minus,
			DebugHotkey.speedDown
		},
		{
			Key.Equal,
			DebugHotkey.speedUp
		},
		{
			Key.F1,
			DebugHotkey.hideRestSite
		},
		{
			Key.F3,
			DebugHotkey.hideEventUi
		},
		{
			Key.F4,
			DebugHotkey.hideProceedButton
		},
		{
			Key.F5,
			DebugHotkey.hideHoverTips
		},
		{
			Key.F6,
			DebugHotkey.hideMpCursors
		},
		{
			Key.F7,
			DebugHotkey.hideMpTargeting
		},
		{
			Key.F9,
			DebugHotkey.hideMpIntents
		},
		{
			Key.F10,
			DebugHotkey.hideMpHealthBars
		},
		{
			Key.U,
			DebugHotkey.unlockCharacters
		}
	};

	public static readonly IReadOnlyList<StringName> remappableKeyboardInputs = new List<StringName>
	{
		MegaInput.select,
		MegaInput.cancel,
		MegaInput.viewMap,
		MegaInput.viewDeckAndTabLeft,
		MegaInput.viewDrawPile,
		MegaInput.viewDiscardPile,
		MegaInput.viewExhaustPileAndTabRight,
		MegaInput.accept,
		MegaInput.peek,
		MegaInput.up,
		MegaInput.down,
		MegaInput.left,
		MegaInput.right,
		MegaInput.selectCard1,
		MegaInput.selectCard2,
		MegaInput.selectCard3,
		MegaInput.selectCard4,
		MegaInput.selectCard5,
		MegaInput.selectCard6,
		MegaInput.selectCard7,
		MegaInput.selectCard8,
		MegaInput.selectCard9,
		MegaInput.selectCard10,
		MegaInput.releaseCard
	};

	public static readonly IReadOnlyList<StringName> remappableControllerInputs = new List<StringName>
	{
		MegaInput.select,
		MegaInput.cancel,
		MegaInput.viewMap,
		MegaInput.topPanel,
		MegaInput.viewDeckAndTabLeft,
		MegaInput.viewDrawPile,
		MegaInput.viewDiscardPile,
		MegaInput.viewExhaustPileAndTabRight,
		MegaInput.accept,
		MegaInput.peek,
		MegaInput.up,
		MegaInput.down,
		MegaInput.left,
		MegaInput.right
	};

	private Dictionary<StringName, Key> _keyboardInputMap = new Dictionary<StringName, Key>();

	private Dictionary<StringName, StringName> _controllerInputMap = new Dictionary<StringName, StringName>();

	public static NInputManager? Instance
	{
		get
		{
			if (NGame.Instance == null)
			{
				return null;
			}
			return NGame.Instance.InputManager;
		}
	}

	private static Dictionary<StringName, Key> DefaultKeyboardInputMap => new Dictionary<StringName, Key>
	{
		{
			MegaInput.accept,
			Key.E
		},
		{
			MegaInput.select,
			Key.Enter
		},
		{
			MegaInput.viewDiscardPile,
			Key.S
		},
		{
			MegaInput.viewDeckAndTabLeft,
			Key.D
		},
		{
			MegaInput.viewExhaustPileAndTabRight,
			Key.X
		},
		{
			MegaInput.viewDrawPile,
			Key.A
		},
		{
			MegaInput.viewMap,
			Key.M
		},
		{
			MegaInput.cancel,
			Key.Escape
		},
		{
			MegaInput.peek,
			Key.Space
		},
		{
			MegaInput.up,
			Key.Up
		},
		{
			MegaInput.down,
			Key.Down
		},
		{
			MegaInput.left,
			Key.Left
		},
		{
			MegaInput.right,
			Key.Right
		},
		{
			MegaInput.pauseAndBack,
			Key.Escape
		},
		{
			MegaInput.selectCard1,
			Key.Key1
		},
		{
			MegaInput.selectCard2,
			Key.Key2
		},
		{
			MegaInput.selectCard3,
			Key.Key3
		},
		{
			MegaInput.selectCard4,
			Key.Key4
		},
		{
			MegaInput.selectCard5,
			Key.Key5
		},
		{
			MegaInput.selectCard6,
			Key.Key6
		},
		{
			MegaInput.selectCard7,
			Key.Key7
		},
		{
			MegaInput.selectCard8,
			Key.Key8
		},
		{
			MegaInput.selectCard9,
			Key.Key9
		},
		{
			MegaInput.selectCard10,
			Key.Key0
		},
		{
			MegaInput.releaseCard,
			Key.Down
		}
	};

	public NControllerManager ControllerManager { get; private set; }

	public override void _EnterTree()
	{
		ControllerManager = GetNode<NControllerManager>("%ControllerManager");
	}

	public override void _Ready()
	{
		ControllerManager.Connect(NControllerManager.SignalName.ControllerTypeChanged, Callable.From(OnControllerTypeChanged));
		TaskHelper.RunSafely(Init());
	}

	private async Task Init()
	{
		await ControllerManager.Init();
		SettingsSave settingsSave = SaveManager.Instance.SettingsSave;
		if (settingsSave.KeyboardMapping.Count > 0)
		{
			_keyboardInputMap = new Dictionary<StringName, Key>();
			foreach (KeyValuePair<string, string> item in settingsSave.KeyboardMapping)
			{
				_keyboardInputMap.Add(item.Key, Enum.Parse<Key>(item.Value));
			}
		}
		else
		{
			_keyboardInputMap = DefaultKeyboardInputMap;
			SaveKeyboardInputMapping();
		}
		if (settingsSave.ControllerMapping.Count > 0 && settingsSave.ControllerMappingType == ControllerManager.ControllerMappingType)
		{
			_controllerInputMap = new Dictionary<StringName, StringName>();
			{
				foreach (KeyValuePair<string, string> item2 in settingsSave.ControllerMapping)
				{
					_controllerInputMap.Add(item2.Key, item2.Value);
				}
				return;
			}
		}
		_controllerInputMap = ControllerManager.GetDefaultControllerInputMap;
		SaveControllerInputMapping();
	}

	public override void _UnhandledKeyInput(InputEvent inputEvent)
	{
		ProcessShortcutKeyInput(inputEvent);
		ProcessDebugKeyInput(inputEvent);
	}

	private void ProcessDebugKeyInput(InputEvent inputEvent)
	{
		if (!(inputEvent is InputEventKey inputEventKey) || NDevConsole.Instance.Visible || !NGame.IsTrailerMode)
		{
			return;
		}
		foreach (KeyValuePair<Key, StringName> item in _debugInputMap)
		{
			if (inputEventKey.Keycode == item.Key)
			{
				InputEventAction inputEventAction = new InputEventAction
				{
					Action = item.Value,
					Pressed = inputEvent.IsPressed()
				};
				Input.ParseInputEvent(inputEventAction);
			}
		}
	}

	private void ProcessShortcutKeyInput(InputEvent inputEvent)
	{
		if (NGame.Instance.Transition.InTransition || !(inputEvent is InputEventKey inputEventKey))
		{
			return;
		}
		foreach (KeyValuePair<StringName, Key> item in _keyboardInputMap)
		{
			if (inputEventKey.Keycode == item.Value && !inputEvent.IsEcho())
			{
				InputEventAction inputEventAction = new InputEventAction
				{
					Action = item.Key,
					Pressed = inputEvent.IsPressed()
				};
				Input.ParseInputEvent(inputEventAction);
			}
		}
	}

	public override void _UnhandledInput(InputEvent inputEvent)
	{
		if (NGame.Instance.Transition.InTransition)
		{
			return;
		}
		foreach (KeyValuePair<StringName, StringName> item in _controllerInputMap)
		{
			if (inputEvent.IsActionPressed(item.Value))
			{
				InputEventAction inputEventAction = new InputEventAction
				{
					Action = item.Key,
					Pressed = true
				};
				Input.ParseInputEvent(inputEventAction);
			}
			else if (inputEvent.IsActionReleased(item.Value))
			{
				InputEventAction inputEventAction2 = new InputEventAction
				{
					Action = item.Key,
					Pressed = false
				};
				Input.ParseInputEvent(inputEventAction2);
			}
		}
	}

	public Key GetShortcutKey(StringName input)
	{
		return _keyboardInputMap[input];
	}

	public Texture2D? GetHotkeyIcon(string hotkey)
	{
		if (_controllerInputMap.TryGetValue(hotkey, out StringName value))
		{
			return ControllerManager.GetHotkeyIcon(value);
		}
		return null;
	}

	public void ModifyShortcutKey(StringName input, Key shortcutKey)
	{
		KeyValuePair<StringName, Key> keyValuePair = _keyboardInputMap.FirstOrDefault<KeyValuePair<StringName, Key>>((KeyValuePair<StringName, Key> kvp) => kvp.Value == shortcutKey && remappableKeyboardInputs.Contains(kvp.Key));
		if (keyValuePair.Key != null)
		{
			Key value = _keyboardInputMap[input];
			_keyboardInputMap[keyValuePair.Key] = value;
		}
		_keyboardInputMap[input] = shortcutKey;
		SaveKeyboardInputMapping();
		EmitSignalInputRebound();
	}

	public void ModifyControllerButton(StringName input, StringName controllerInput)
	{
		KeyValuePair<StringName, StringName> keyValuePair = _controllerInputMap.FirstOrDefault<KeyValuePair<StringName, StringName>>((KeyValuePair<StringName, StringName> kvp) => kvp.Value == controllerInput && remappableControllerInputs.Contains(kvp.Key));
		if (keyValuePair.Key != null)
		{
			StringName value = _controllerInputMap[input];
			_controllerInputMap[keyValuePair.Key] = value;
		}
		_controllerInputMap[input] = controllerInput;
		SaveControllerInputMapping();
		EmitSignalInputRebound();
	}

	public void ResetToDefaults()
	{
		_keyboardInputMap = DefaultKeyboardInputMap;
		_controllerInputMap = ControllerManager.GetDefaultControllerInputMap;
		SaveControllerInputMapping();
		SaveKeyboardInputMapping();
		EmitSignalInputRebound();
	}

	public void ResetToDefaultControllerMapping()
	{
		_controllerInputMap = ControllerManager.GetDefaultControllerInputMap;
		SaveControllerInputMapping();
		EmitSignalInputRebound();
	}

	private void OnControllerTypeChanged()
	{
		if (ControllerManager.ControllerMappingType != SaveManager.Instance.SettingsSave.ControllerMappingType)
		{
			_controllerInputMap = ControllerManager.GetDefaultControllerInputMap;
			SaveControllerInputMapping();
			EmitSignalInputRebound();
		}
	}

	private void SaveControllerInputMapping()
	{
		Dictionary<string, string> dictionary = new Dictionary<string, string>();
		foreach (KeyValuePair<StringName, StringName> item in _controllerInputMap)
		{
			dictionary.Add(item.Key.ToString(), item.Value.ToString());
		}
		SaveManager.Instance.SettingsSave.ControllerMappingType = ControllerManager.ControllerMappingType;
		SaveManager.Instance.SettingsSave.ControllerMapping = dictionary;
		SaveManager.Instance.SaveSettings();
	}

	private void SaveKeyboardInputMapping()
	{
		Dictionary<string, string> dictionary = new Dictionary<string, string>();
		foreach (KeyValuePair<StringName, Key> item in _keyboardInputMap)
		{
			dictionary.Add(item.Key.ToString(), item.Value.ToString());
		}
		SaveManager.Instance.SettingsSave.KeyboardMapping = dictionary;
		SaveManager.Instance.SaveSettings();
	}
}
