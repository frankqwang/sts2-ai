using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Multiplayer;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NResetGraphicsButton : NSettingsButton
{
	public override void _Ready()
	{
		ConnectSignals();
		base.PivotOffset = base.Size * 0.5f;
		GetNode<MegaLabel>("Label").SetTextAutoSize(new LocString("settings_ui", "RESET_SETTINGS_BUTTON").GetFormattedText());
	}

	private async Task ResetSettingsAfterConfirmation()
	{
		NGenericPopup nGenericPopup = NGenericPopup.Create();
		NModalContainer.Instance.Add(nGenericPopup);
		if (!(await nGenericPopup.WaitForConfirmation(new LocString("settings_ui", "RESET_GRAPHICS_CONFIRMATION.body"), new LocString("settings_ui", "RESET_CONFIRMATION.header"), new LocString("main_menu_ui", "GENERIC_POPUP.cancel"), new LocString("main_menu_ui", "GENERIC_POPUP.confirm"))))
		{
			return;
		}
		Log.Info("Player reset graphics settings");
		SettingsSave settingsSave = SaveManager.Instance.SettingsSave;
		settingsSave.FpsLimit = 60;
		settingsSave.WindowPosition = new Vector2I(-1, -1);
		settingsSave.WindowSize = new Vector2I(1920, 1080);
		settingsSave.Fullscreen = true;
		settingsSave.AspectRatioSetting = AspectRatioSetting.SixteenByNine;
		settingsSave.TargetDisplay = -1;
		settingsSave.ResizeWindows = true;
		settingsSave.VSync = VSyncType.Adaptive;
		settingsSave.Msaa = 2;
		NGame.Instance.ApplyDisplaySettings();
		NSettingsPanel ancestorOfType = this.GetAncestorOfType<NSettingsPanel>();
		foreach (IResettableSettingNode item in ancestorOfType.GetChildrenRecursive<IResettableSettingNode>())
		{
			item.SetFromSettings();
		}
	}

	protected override void OnRelease()
	{
		base.OnRelease();
		TaskHelper.RunSafely(ResetSettingsAfterConfirmation());
	}
}
