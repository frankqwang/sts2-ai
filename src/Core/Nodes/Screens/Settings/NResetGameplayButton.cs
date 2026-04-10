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

public partial class NResetGameplayButton : NSettingsButton
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
		if (!(await nGenericPopup.WaitForConfirmation(new LocString("settings_ui", "RESET_GAMEPLAY_CONFIRMATION.body"), new LocString("settings_ui", "RESET_CONFIRMATION.header"), new LocString("main_menu_ui", "GENERIC_POPUP.cancel"), new LocString("main_menu_ui", "GENERIC_POPUP.confirm"))))
		{
			return;
		}
		Log.Info("Player reset general settings");
		SettingsSave settingsSave = SaveManager.Instance.SettingsSave;
		settingsSave.LimitFpsInBackground = true;
		settingsSave.SkipIntroLogo = false;
		PrefsSave prefsSave = SaveManager.Instance.PrefsSave;
		prefsSave.ScreenShakeOptionIndex = 2;
		prefsSave.FastMode = FastModeType.Normal;
		prefsSave.ShowRunTimer = false;
		prefsSave.ShowCardIndices = false;
		prefsSave.IsLongPressEnabled = false;
		prefsSave.UploadData = true;
		prefsSave.TextEffectsEnabled = true;
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
