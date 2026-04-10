using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NInputSettingsEntry : NButton
{
	private static readonly Dictionary<StringName, string> _commandToLocTitle = new Dictionary<StringName, string>
	{
		{
			MegaInput.accept,
			"endTurn"
		},
		{
			MegaInput.select,
			"confirmCard"
		},
		{
			MegaInput.viewDiscardPile,
			"viewDiscard"
		},
		{
			MegaInput.viewDrawPile,
			"viewDraw"
		},
		{
			MegaInput.viewDeckAndTabLeft,
			"viewDeck"
		},
		{
			MegaInput.viewExhaustPileAndTabRight,
			"viewExhaust"
		},
		{
			MegaInput.viewMap,
			"viewMap"
		},
		{
			MegaInput.cancel,
			"cancel"
		},
		{
			MegaInput.peek,
			"peek"
		},
		{
			MegaInput.up,
			"up"
		},
		{
			MegaInput.topPanel,
			"topPanel"
		},
		{
			MegaInput.down,
			"down"
		},
		{
			MegaInput.left,
			"left"
		},
		{
			MegaInput.right,
			"right"
		},
		{
			MegaInput.selectCard1,
			"selectCard1"
		},
		{
			MegaInput.selectCard2,
			"selectCard2"
		},
		{
			MegaInput.selectCard3,
			"selectCard3"
		},
		{
			MegaInput.selectCard4,
			"selectCard4"
		},
		{
			MegaInput.selectCard5,
			"selectCard5"
		},
		{
			MegaInput.selectCard6,
			"selectCard6"
		},
		{
			MegaInput.selectCard7,
			"selectCard7"
		},
		{
			MegaInput.selectCard8,
			"selectCard8"
		},
		{
			MegaInput.selectCard9,
			"selectCard9"
		},
		{
			MegaInput.selectCard10,
			"selectCard10"
		},
		{
			MegaInput.releaseCard,
			"releaseCard"
		}
	};

	private const string _scenePath = "res://scenes/screens/settings_screen/input_settings_entry.tscn";

	private Control _bg;

	private MegaRichTextLabel _inputLabel;

	private MegaRichTextLabel _keyBindingLabel;

	private TextureRect _controllerBindingIcon;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>("res://scenes/screens/settings_screen/input_settings_entry.tscn");

	public StringName InputName { get; private set; }

	public static NInputSettingsEntry Create(string commandName)
	{
		NInputSettingsEntry nInputSettingsEntry = ResourceLoader.Load<PackedScene>("res://scenes/screens/settings_screen/input_settings_entry.tscn", null, ResourceLoader.CacheMode.Reuse).Instantiate<NInputSettingsEntry>(PackedScene.GenEditState.Disabled);
		nInputSettingsEntry.InputName = commandName;
		return nInputSettingsEntry;
	}

	public override void _Ready()
	{
		ConnectSignals();
		_inputLabel = GetNode<MegaRichTextLabel>("%InputLabel");
		_keyBindingLabel = GetNode<MegaRichTextLabel>("%KeyBindingInputLabel");
		_controllerBindingIcon = GetNode<TextureRect>("%ControllerBindingIcon");
		_bg = GetNode<Control>("%Bg");
		string text = _commandToLocTitle[InputName];
		_inputLabel.Text = new LocString("settings_ui", "INPUT_SETTINGS.INPUT_TITLE." + text).GetFormattedText();
		NInputManager.Instance.Connect(NInputManager.SignalName.InputRebound, Callable.From(UpdateInput));
		NControllerManager.Instance.Connect(NControllerManager.SignalName.ControllerDetected, Callable.From(UpdateInput));
		NControllerManager.Instance.Connect(NControllerManager.SignalName.MouseDetected, Callable.From(UpdateInput));
		Connect(CanvasItem.SignalName.VisibilityChanged, Callable.From(UpdateInput));
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		NInputManager.Instance.Disconnect(NInputManager.SignalName.InputRebound, Callable.From(UpdateInput));
		NControllerManager.Instance.Disconnect(NControllerManager.SignalName.ControllerDetected, Callable.From(UpdateInput));
		NControllerManager.Instance.Disconnect(NControllerManager.SignalName.MouseDetected, Callable.From(UpdateInput));
		Disconnect(CanvasItem.SignalName.VisibilityChanged, Callable.From(UpdateInput));
	}

	private void UpdateInput()
	{
		if (IsVisibleInTree())
		{
			if (NInputManager.remappableKeyboardInputs.Contains(InputName))
			{
				_keyBindingLabel.Text = NInputManager.Instance.GetShortcutKey(InputName).ToString();
			}
			else
			{
				_keyBindingLabel.Text = "";
			}
			if (NInputManager.remappableControllerInputs.Contains(InputName))
			{
				_controllerBindingIcon.Texture = NInputManager.Instance.GetHotkeyIcon(InputName);
			}
			_controllerBindingIcon.Modulate = (NControllerManager.Instance.ShouldAllowControllerRebinding ? Colors.White : new Color(1f, 1f, 1f, 0.15f));
		}
	}

	protected override void OnFocus()
	{
		_bg.Visible = true;
	}

	protected override void OnUnfocus()
	{
		_bg.Visible = false;
	}
}
