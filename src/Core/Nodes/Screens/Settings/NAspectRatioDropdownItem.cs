using System;
using Godot;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Settings;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NAspectRatioDropdownItem : NDropdownItem
{
	public AspectRatioSetting aspectRatioSetting;

	public void Init(AspectRatioSetting setAspectRatioSetting)
	{
		aspectRatioSetting = setAspectRatioSetting;
		switch (aspectRatioSetting)
		{
		case AspectRatioSetting.Auto:
			_label.SetTextAutoSize(new LocString("settings_ui", "ASPECT_RATIO_AUTO").GetFormattedText());
			break;
		case AspectRatioSetting.FourByThree:
			_label.SetTextAutoSize(new LocString("settings_ui", "ASPECT_RATIO_FOUR_BY_THREE").GetFormattedText());
			break;
		case AspectRatioSetting.SixteenByTen:
			_label.SetTextAutoSize(new LocString("settings_ui", "ASPECT_RATIO_SIXTEEN_BY_TEN").GetFormattedText());
			break;
		case AspectRatioSetting.SixteenByNine:
			_label.SetTextAutoSize(new LocString("settings_ui", "ASPECT_RATIO_SIXTEEN_BY_NINE").GetFormattedText());
			break;
		case AspectRatioSetting.TwentyOneByNine:
			_label.SetTextAutoSize(new LocString("settings_ui", "ASPECT_RATIO_TWENTY_ONE_BY_NINE").GetFormattedText());
			break;
		default:
			throw new ArgumentOutOfRangeException($"Invalid Aspect Ratio: {aspectRatioSetting}");
		}
	}
}
