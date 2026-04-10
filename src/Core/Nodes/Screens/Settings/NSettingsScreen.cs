using System;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.Capstones;
using MegaCrit.Sts2.Core.Nodes.Screens.FeedbackScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Nodes.Screens.ModdingScreen;
using MegaCrit.Sts2.Core.Platform;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NSettingsScreen : NSubmenu
{
	[Signal]
	public delegate void SettingsClosedEventHandler();

	[Signal]
	public delegate void SettingsOpenedEventHandler();

	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/settings_screen");

	private NSettingsTabManager _settingsTabManager;

	private NOpenFeedbackScreenButton _feedbackScreenButton;

	private NOpenModdingScreenButton _moddingScreenButton;

	private NSettingsToast _toast;

	private bool _isInRun;

	public static readonly Vector2 settingTipsOffset = new Vector2(1012f, -60f);

	public static string[] AssetPaths => new string[2] { _scenePath, "res://images/ui/language_warning.png" };

	protected override Control? InitialFocusedControl => _settingsTabManager.DefaultFocusedControl;

	public void SetIsInRun(bool isInRun)
	{
		_isInRun = isInRun;
	}

	public override void _Ready()
	{
		ConnectSignals();
		_settingsTabManager = GetNode<NSettingsTabManager>("%SettingsTabManager");
		_feedbackScreenButton = GetNode<NOpenFeedbackScreenButton>("%FeedbackButton");
		_moddingScreenButton = GetNode<NOpenModdingScreenButton>("%ModdingButton");
		_toast = GetNode<NSettingsToast>("%Toast");
		LocalizeLabels();
		base.ProcessMode = (ProcessModeEnum)(base.Visible ? 0 : 4);
		_settingsTabManager.Connect(NSettingsTabManager.SignalName.TabChanged, Callable.From(OnSettingsTabChanged));
		_moddingScreenButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OpenModdingScreen));
		_feedbackScreenButton.Connect(NClickableControl.SignalName.Released, Callable.From((Action<NButton>)OpenFeedbackScreen));
		if (SaveManager.Instance.SettingsSave.ModSettings != null && ModManager.AllMods.Count > 0)
		{
			GetNode<Control>("%Modding").Visible = true;
			GetNode<Control>("%ModdingDivider").Visible = true;
		}
		if (PlatformUtil.GetSupportedWindowMode() == SupportedWindowMode.FullscreenOnly)
		{
			GetNode<Control>("%Fullscreen").Visible = false;
			GetNode<Control>("%FullscreenDivider").Visible = false;
		}
		if (RunManager.Instance.IsInProgress)
		{
			GetNode<Node>("%LanguageLine").GetNode<MegaRichTextLabel>("Label").Modulate = StsColors.gray;
			NLanguageDropdown node = GetNode<NLanguageDropdown>("%LanguageDropdown");
			node.Modulate = StsColors.gray;
			node.Disable();
			_moddingScreenButton.Disable();
		}
	}

	public void ShowToast(LocString locString)
	{
		_toast.Show(locString);
	}

	private void OnSettingsTabChanged()
	{
	}

	private void LocalizeLabels()
	{
		Node content = GetNode<NSettingsPanel>("%GeneralSettings").Content;
		LocHelper(content.GetNode<Node>("LanguageLine"), new LocString("settings_ui", _isInRun ? "LANGUAGE_IN_RUN" : "LANGUAGE"));
		LocHelper(content.GetNode<Node>("FastMode"), new LocString("settings_ui", "FASTMODE"));
		LocHelper(content.GetNode<Node>("Screenshake"), new LocString("settings_ui", "SCREENSHAKE"));
		LocHelper(content.GetNode<Node>("CommonTooltips"), new LocString("settings_ui", "COMMON_TOOLTIPS"));
		LocHelper(content.GetNode<Node>("ShowRunTimer"), new LocString("settings_ui", "SHOW_RUN_TIMER_HEADER"));
		LocHelper(content.GetNode<Node>("ShowHandCardCount"), new LocString("settings_ui", "SHOW_HAND_CARD_COUNT_HEADER"));
		LocHelper(content.GetNode<Node>("LongPressConfirmations"), new LocString("settings_ui", "LONG_PRESS_CONFIRMATION_HEADER"));
		LocHelper(content.GetNode<Node>("SkipIntroLogo"), new LocString("settings_ui", "SKIP_INTRO_LOGO_HEADER"));
		LocHelper(content.GetNode<Node>("LimitFpsInBackground"), new LocString("settings_ui", "LIMIT_FPS_IN_BACKGROUND_HEADER"));
		LocHelper(content.GetNode<Node>("UploadGameplayData"), new LocString("settings_ui", "UPLOAD_GAMEPLAY_DATA"));
		LocHelper(content.GetNode<Node>("TextEffects"), new LocString("settings_ui", "TEXT_EFFECTS"));
		LocHelper(content.GetNode<Node>("SendFeedback"), new LocString("settings_ui", "SEND_FEEDBACK"));
		LocHelper(content.GetNode<Node>("ResetTutorials"), new LocString("settings_ui", "TUTORIAL_RESET"));
		LocHelper(content.GetNode<Node>("Credits"), new LocString("settings_ui", "CREDITS"));
		LocHelper(content.GetNode<Node>("ResetGameplay"), new LocString("settings_ui", "RESET_DEFAULT"));
		Node content2 = GetNode<NSettingsPanel>("%GraphicsSettings").Content;
		LocHelper(content2.GetNode<Node>("Fullscreen"), new LocString("settings_ui", "FULLSCREEN"));
		LocHelper(content2.GetNode<Node>("DisplaySelection"), new LocString("settings_ui", "DISPLAY_SELECTION"));
		LocHelper(content2.GetNode<Node>("WindowedResolution"), new LocString("settings_ui", "RESOLUTION"));
		LocHelper(content2.GetNode<Node>("AspectRatio"), new LocString("settings_ui", "ASPECT_RATIO"));
		LocHelper(content2.GetNode<Node>("WindowResizing"), new LocString("settings_ui", "WINDOW_RESIZING"));
		LocHelper(content2.GetNode<Node>("VSync"), new LocString("settings_ui", "VSYNC"));
		LocHelper(content2.GetNode<Node>("MaxFps"), new LocString("settings_ui", "FPS_CAP"));
		LocHelper(content2.GetNode<Node>("Msaa"), new LocString("settings_ui", "MSAA"));
		LocHelper(content2.GetNode<Node>("ResetGraphics"), new LocString("settings_ui", "RESET_DEFAULT"));
		Node content3 = GetNode<NSettingsPanel>("%SoundSettings").Content;
		LocHelper(content3.GetNode<Node>("MasterVolume"), new LocString("settings_ui", "MASTER_VOLUME"));
		LocHelper(content3.GetNode<Node>("BgmVolume"), new LocString("settings_ui", "MUSIC_VOLUME"));
		LocHelper(content3.GetNode<Node>("SfxVolume"), new LocString("settings_ui", "SFX_VOLUME"));
		LocHelper(content3.GetNode<Node>("AmbienceVolume"), new LocString("settings_ui", "AMBIENCE_VOLUME"));
		LocHelper(content3.GetNode<Node>("MuteIfBackground"), new LocString("settings_ui", "BACKGROUND_MUTE"));
	}

	private static void LocHelper(Node settingsLineNode, LocString locString)
	{
		settingsLineNode.GetNode<MegaRichTextLabel>("Label").Text = locString.GetFormattedText();
	}

	private void OpenModdingScreen(NButton _)
	{
		_stack.PushSubmenuType<NModdingScreen>();
	}

	private void OpenFeedbackScreen(NButton _)
	{
		_lastFocusedControl = _feedbackScreenButton;
		TaskHelper.RunSafely(OpenFeedbackScreen());
	}

	public async Task OpenFeedbackScreen()
	{
		Log.Info("Opening feedback screen");
		base.Visible = false;
		NGame.Instance.MainMenu?.DisableBackstopInstantly();
		NCapstoneContainer.Instance?.DisableBackstopInstantly();
		NRun.Instance?.GlobalUi.RelicInventory.ShowImmediately();
		NRun.Instance?.GlobalUi.MultiplayerPlayerContainer.ShowImmediately();
		await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		Image image = GetViewport().GetTexture().GetImage();
		base.Visible = true;
		NRun.Instance?.GlobalUi.RelicInventory.HideImmediately();
		NRun.Instance?.GlobalUi.MultiplayerPlayerContainer.HideImmediately();
		NGame.Instance.MainMenu?.EnableBackstopInstantly();
		NCapstoneContainer.Instance?.EnableBackstopInstantly();
		NSendFeedbackScreen feedbackScreen = NGame.Instance.FeedbackScreen;
		feedbackScreen.SetScreenshot(image);
		NGame.Instance.FeedbackScreen.Open();
	}

	public override void OnSubmenuOpened()
	{
		base.OnSubmenuOpened();
		base.ProcessMode = ProcessModeEnum.Inherit;
		EmitSignal(SignalName.SettingsOpened);
		_settingsTabManager.ResetTabs();
		_settingsTabManager.Enable();
	}

	public override void OnSubmenuClosed()
	{
		base.OnSubmenuClosed();
		base.ProcessMode = ProcessModeEnum.Disabled;
		SaveManager.Instance.SaveSettings();
		SaveManager.Instance.SavePrefsFile();
		EmitSignal(SignalName.SettingsClosed);
		_settingsTabManager.Disable();
	}

	protected override void OnSubmenuHidden()
	{
		base.OnSubmenuClosed();
		base.ProcessMode = ProcessModeEnum.Disabled;
		EmitSignal(SignalName.SettingsClosed);
	}

	protected override void OnSubmenuShown()
	{
		base.OnSubmenuShown();
		base.ProcessMode = ProcessModeEnum.Inherit;
		EmitSignal(SignalName.SettingsOpened);
	}
}
