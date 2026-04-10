using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NInputSettingsPanel : NSettingsPanel
{
	private float _minPadding = 50f;

	private NInputSettingsEntry? _listeningEntry;

	private NButton _resetToDefaultButton;

	private MegaRichTextLabel _commandHeader;

	private MegaRichTextLabel _keyboardHeader;

	private MegaRichTextLabel _controllerHeader;

	private MegaRichTextLabel _steamInputPrompt;

	public override void _Ready()
	{
		base._Ready();
		_resetToDefaultButton = GetNode<NButton>("%ResetToDefaultButton");
		_commandHeader = GetNode<MegaRichTextLabel>("%CommandHeader");
		_keyboardHeader = GetNode<MegaRichTextLabel>("%KeyboardHeader");
		_controllerHeader = GetNode<MegaRichTextLabel>("%ControllerHeader");
		_steamInputPrompt = GetNode<MegaRichTextLabel>("%SteamInputPrompt");
		_resetToDefaultButton.Connect(NClickableControl.SignalName.Released, Callable.From<NClickableControl>(delegate
		{
			NInputManager.Instance.ResetToDefaults();
		}));
		GetViewport().Connect(Viewport.SignalName.SizeChanged, Callable.From(OnViewportSizeChange));
		_commandHeader.Text = new LocString("settings_ui", "INPUT_SETTINGS.COMMAND_HEADER").GetFormattedText();
		_keyboardHeader.Text = new LocString("settings_ui", "INPUT_SETTINGS.KEYBOARD_HEADER").GetFormattedText();
		_controllerHeader.Text = new LocString("settings_ui", "INPUT_SETTINGS.CONTROLLER_HEADER").GetFormattedText();
		_steamInputPrompt.Text = new LocString("settings_ui", "INPUT_SETTINGS.STEAM_INPUT_DETECTED").GetFormattedText();
		IReadOnlyList<StringName> readOnlyList = NInputManager.remappableControllerInputs.Concat(NInputManager.remappableKeyboardInputs).Distinct().ToList();
		List<NInputSettingsEntry> list = base.Content.GetChildren().OfType<NInputSettingsEntry>().ToList();
		foreach (StringName item in readOnlyList)
		{
			NInputSettingsEntry entry = NInputSettingsEntry.Create(item);
			entry.Connect(NClickableControl.SignalName.Released, Callable.From<NClickableControl>(delegate
			{
				SetAsListeningEntry(entry);
			}));
			base.Content.AddChildSafely(entry);
			list.Add(entry);
		}
		for (int num = 0; num < list.Count; num++)
		{
			list[num].FocusNeighborLeft = list[num].GetPath();
			list[num].FocusNeighborRight = list[num].GetPath();
			list[num].FocusNeighborTop = ((num > 0) ? list[num - 1].GetPath() : list[num].GetPath());
			list[num].FocusNeighborBottom = ((num < list.Count - 1) ? list[num + 1].GetPath() : list[num].GetPath());
		}
		_resetToDefaultButton.FocusNeighborLeft = _resetToDefaultButton.GetPath();
		_resetToDefaultButton.FocusNeighborRight = _resetToDefaultButton.GetPath();
		_resetToDefaultButton.FocusNeighborTop = _resetToDefaultButton.GetPath();
		_resetToDefaultButton.FocusNeighborBottom = list[0].GetPath();
		list[0].FocusNeighborTop = _resetToDefaultButton.GetPath();
		_firstControl = base.Content.GetChildren().OfType<NInputSettingsEntry>().First();
	}

	private async Task RefreshSize()
	{
		await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		Vector2 size = GetParent<Control>().Size;
		Vector2 minimumSize = base.Content.GetMinimumSize();
		if (minimumSize.Y + _minPadding >= size.Y)
		{
			base.Size = new Vector2(base.Content.Size.X, minimumSize.Y + size.Y * 0.4f);
		}
	}

	private void OnViewportSizeChange()
	{
		TaskHelper.RunSafely(RefreshSize());
	}

	protected override void OnVisibilityChange()
	{
		base.OnVisibilityChange();
		_listeningEntry = null;
		_steamInputPrompt.Visible = !NControllerManager.Instance.ShouldAllowControllerRebinding;
		TaskHelper.RunSafely(RefreshSize());
	}

	private void SetAsListeningEntry(NInputSettingsEntry entry)
	{
		_listeningEntry = entry;
	}

	public override void _UnhandledKeyInput(InputEvent inputEvent)
	{
		if (_listeningEntry != null && NInputManager.remappableKeyboardInputs.Contains(_listeningEntry.InputName) && inputEvent is InputEventKey inputEventKey)
		{
			NInputManager.Instance.ModifyShortcutKey(_listeningEntry.InputName, inputEventKey.Keycode);
			GetViewport()?.SetInputAsHandled();
			_listeningEntry = null;
		}
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (_listeningEntry == null)
		{
			return;
		}
		StringName[] allControllerInputs = Controller.AllControllerInputs;
		foreach (StringName stringName in allControllerInputs)
		{
			if (inputEvent.IsActionReleased(stringName))
			{
				if (NInputManager.remappableControllerInputs.Contains(_listeningEntry.InputName) && NControllerManager.Instance.ShouldAllowControllerRebinding)
				{
					NInputManager.Instance.ModifyControllerButton(_listeningEntry.InputName, stringName);
				}
				GetViewport()?.SetInputAsHandled();
				_listeningEntry = null;
				break;
			}
		}
	}
}
