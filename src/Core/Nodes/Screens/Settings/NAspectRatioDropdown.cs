using System;
using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NAspectRatioDropdown : NSettingsDropdown, IResettableSettingNode
{
	private AspectRatioSetting _currentAspectRatioSetting;

	private static string AspectRatioDropdownItemScenePath => SceneHelper.GetScenePath("ui/aspect_ratio_dropdown_item");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(AspectRatioDropdownItemScenePath);

	public override void _Ready()
	{
		ConnectSignals();
		ClearDropdownItems();
		AddDropdownItem(AspectRatioSetting.Auto);
		AddDropdownItem(AspectRatioSetting.FourByThree);
		AddDropdownItem(AspectRatioSetting.SixteenByTen);
		AddDropdownItem(AspectRatioSetting.SixteenByNine);
		AddDropdownItem(AspectRatioSetting.TwentyOneByNine);
		_dropdownItems.GetParent<NDropdownContainer>().RefreshLayout();
		SetFromSettings();
	}

	public void SetFromSettings()
	{
		_currentAspectRatioSetting = SaveManager.Instance.SettingsSave.AspectRatioSetting;
		_currentOptionLabel.SetTextAutoSize(GetAspectRatioSettingString(_currentAspectRatioSetting));
	}

	private void AddDropdownItem(AspectRatioSetting aspectRatioSetting)
	{
		NAspectRatioDropdownItem nAspectRatioDropdownItem = ResourceLoader.Load<PackedScene>(AspectRatioDropdownItemScenePath, null, ResourceLoader.CacheMode.Reuse).Instantiate<NAspectRatioDropdownItem>(PackedScene.GenEditState.Disabled);
		_dropdownItems.AddChildSafely(nAspectRatioDropdownItem);
		nAspectRatioDropdownItem.Connect(NDropdownItem.SignalName.Selected, Callable.From<NDropdownItem>(OnDropdownItemSelected));
		nAspectRatioDropdownItem.Init(aspectRatioSetting);
	}

	private void OnDropdownItemSelected(NDropdownItem nDropdownItem)
	{
		NAspectRatioDropdownItem nAspectRatioDropdownItem = (NAspectRatioDropdownItem)nDropdownItem;
		if (nAspectRatioDropdownItem.aspectRatioSetting != _currentAspectRatioSetting)
		{
			CloseDropdown();
			SaveManager.Instance.SettingsSave.AspectRatioSetting = nAspectRatioDropdownItem.aspectRatioSetting;
			SetFromSettings();
			NGame.Instance.ApplyDisplaySettings();
		}
	}

	private static string GetAspectRatioSettingString(AspectRatioSetting aspectRatioSettingString)
	{
		return aspectRatioSettingString switch
		{
			AspectRatioSetting.Auto => new LocString("settings_ui", "ASPECT_RATIO_AUTO").GetFormattedText(), 
			AspectRatioSetting.FourByThree => new LocString("settings_ui", "ASPECT_RATIO_FOUR_BY_THREE").GetFormattedText(), 
			AspectRatioSetting.SixteenByTen => new LocString("settings_ui", "ASPECT_RATIO_SIXTEEN_BY_TEN").GetFormattedText(), 
			AspectRatioSetting.SixteenByNine => new LocString("settings_ui", "ASPECT_RATIO_SIXTEEN_BY_NINE").GetFormattedText(), 
			AspectRatioSetting.TwentyOneByNine => new LocString("settings_ui", "ASPECT_RATIO_TWENTY_ONE_BY_NINE").GetFormattedText(), 
			_ => throw new ArgumentOutOfRangeException($"Invalid Aspect Ratio: {aspectRatioSettingString}"), 
		};
	}
}
