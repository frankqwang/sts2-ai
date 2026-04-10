using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Platform;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NDisplayDropdown : NSettingsDropdown
{
	[Export(PropertyHint.None, "")]
	private PackedScene _dropdownItemScene;

	private static readonly LocString _optionString = new LocString("settings_ui", "DISPLAY_DROPDOWN_OPTION");

	private int _currentDisplayIndex = -1;

	public override void _Ready()
	{
		ConnectSignals();
		NGame.Instance.Connect(NGame.SignalName.WindowChange, Callable.From<bool>(OnWindowChange));
		OnWindowChange(SaveManager.Instance.SettingsSave.AspectRatioSetting == AspectRatioSetting.Auto);
		ClearDropdownItems();
		for (int i = 0; i < DisplayServer.GetScreenCount(); i++)
		{
			NDisplayDropdownItem nDisplayDropdownItem = _dropdownItemScene.Instantiate<NDisplayDropdownItem>(PackedScene.GenEditState.Disabled);
			_dropdownItems.AddChildSafely(nDisplayDropdownItem);
			nDisplayDropdownItem.Connect(NDropdownItem.SignalName.Selected, Callable.From<NDropdownItem>(OnDropdownItemSelected));
			nDisplayDropdownItem.Init(i);
		}
		_dropdownItems.GetParent<NDropdownContainer>().RefreshLayout();
	}

	public override void _Notification(int what)
	{
		if ((long)what == 1012 && IsNodeReady())
		{
			OnWindowChange(SaveManager.Instance.SettingsSave.AspectRatioSetting == AspectRatioSetting.Auto);
		}
	}

	private void OnWindowChange(bool _)
	{
		long num = GetWindow().CurrentScreen;
		if (num != _currentDisplayIndex)
		{
			_currentDisplayIndex = (int)num;
			_optionString.Add("MonitorIndex", _currentDisplayIndex);
			_currentOptionLabel.SetTextAutoSize(_optionString.GetFormattedText());
			SaveManager.Instance.SettingsSave.TargetDisplay = _currentDisplayIndex;
			NResolutionDropdown? instance = NResolutionDropdown.Instance;
			if (instance != null && instance.IsNodeReady())
			{
				NResolutionDropdown.Instance.RefreshCurrentlySelectedResolution();
				NResolutionDropdown.Instance.PopulateDropdownItems();
			}
		}
	}

	private void OnDropdownItemSelected(NDropdownItem nDropdownItem)
	{
		SettingsSave settingsSave = SaveManager.Instance.SettingsSave;
		NDisplayDropdownItem nDisplayDropdownItem = (NDisplayDropdownItem)nDropdownItem;
		if (nDisplayDropdownItem.displayIndex != _currentDisplayIndex)
		{
			CloseDropdown();
			settingsSave.TargetDisplay = nDisplayDropdownItem.displayIndex;
			if (!settingsSave.Fullscreen && !PlatformUtil.GetSupportedWindowMode().ShouldForceFullscreen())
			{
				Vector2I vector2I = DisplayServer.ScreenGetSize(nDisplayDropdownItem.displayIndex);
				settingsSave.WindowPosition = vector2I / 8;
				settingsSave.WindowSize = new Vector2I((int)((float)vector2I.X * 0.75f), (int)((float)vector2I.Y * 0.75f));
			}
			NResolutionDropdown.Instance.RefreshCurrentlySelectedResolution();
			NResolutionDropdown.Instance.PopulateDropdownItems();
			NGame.Instance.ApplyDisplaySettings();
		}
	}
}
