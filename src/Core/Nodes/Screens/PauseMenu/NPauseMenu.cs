using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Nodes.Audio;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.Capstones;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Nodes.Screens.Settings;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.PauseMenu;

public partial class NPauseMenu : NSubmenu
{
	private static readonly LocString _pausedLoc = new LocString("gameplay_ui", "PAUSE_MENU.PAUSED");

	private static readonly LocString _resumeLoc = new LocString("gameplay_ui", "PAUSE_MENU.RESUME");

	private static readonly LocString _settingsLoc = new LocString("gameplay_ui", "PAUSE_MENU.SETTINGS");

	private static readonly LocString _compendiumLoc = new LocString("gameplay_ui", "PAUSE_MENU.COMPENDIUM");

	private static readonly LocString _giveUpLoc = new LocString("gameplay_ui", "PAUSE_MENU.GIVE_UP");

	private static readonly LocString _disconnectLoc = new LocString("gameplay_ui", "PAUSE_MENU.DISCONNECT");

	private static readonly LocString _saveAndQuitLoc = new LocString("gameplay_ui", "PAUSE_MENU.SAVE_AND_QUIT");

	private NBackButton _backButton;

	private Control _buttonContainer;

	private NPauseMenuButton _resumeButton;

	private NPauseMenuButton _settingsButton;

	private NPauseMenuButton _compendiumButton;

	private NPauseMenuButton _giveUpButton;

	private NPauseMenuButton _disconnectButton;

	private NPauseMenuButton _saveAndQuitButton;

	private MegaLabel _pausedLabel;

	private IRunState _runState;

	protected override Control InitialFocusedControl => _resumeButton;

	private NPauseMenuButton[] Buttons => new NPauseMenuButton[6] { _resumeButton, _settingsButton, _compendiumButton, _giveUpButton, _disconnectButton, _saveAndQuitButton };

	public bool UseSharedBackstop => true;

	public NetScreenType ScreenType => NetScreenType.PauseMenu;

	public override void _Ready()
	{
		ConnectSignals();
		_buttonContainer = GetNode<Control>("%ButtonContainer");
		_resumeButton = _buttonContainer.GetNode<NPauseMenuButton>("Resume");
		_settingsButton = _buttonContainer.GetNode<NPauseMenuButton>("Settings");
		_compendiumButton = _buttonContainer.GetNode<NPauseMenuButton>("Compendium");
		_giveUpButton = _buttonContainer.GetNode<NPauseMenuButton>("GiveUp");
		_disconnectButton = _buttonContainer.GetNode<NPauseMenuButton>("Disconnect");
		_saveAndQuitButton = _buttonContainer.GetNode<NPauseMenuButton>("SaveAndQuit");
		_backButton = GetNode<NBackButton>("%BackButton");
		_pausedLabel = GetNode<MegaLabel>("%PausedText/Label");
		RefreshLabels();
		_resumeButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnBackOrResumeButtonPressed));
		_settingsButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnSettingsButtonPressed));
		_compendiumButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnCompendiumButtonPressed));
		_giveUpButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnGiveUpButtonPressed));
		_disconnectButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnDisconnectButtonPressed));
		_saveAndQuitButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnSaveAndQuitButtonPressed));
		_backButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnBackOrResumeButtonPressed));
		_backButton.Disable();
		_giveUpButton.Visible = RunManager.Instance.NetService.Type != NetGameType.Client;
		_saveAndQuitButton.Visible = RunManager.Instance.NetService.Type != NetGameType.Client;
		_disconnectButton.Visible = RunManager.Instance.NetService.Type == NetGameType.Client;
		for (int i = 0; i < Buttons.Length; i++)
		{
			Buttons[i].FocusNeighborLeft = Buttons[i].GetPath();
			Buttons[i].FocusNeighborRight = Buttons[i].GetPath();
			Buttons[i].FocusNeighborTop = ((i > 0) ? Buttons[i - 1].GetPath() : Buttons[i].GetPath());
			Buttons[i].FocusNeighborBottom = ((i < Buttons.Length - 1) ? Buttons[i + 1].GetPath() : Buttons[i].GetPath());
		}
	}

	private void RefreshLabels()
	{
		_pausedLabel.SetTextAutoSize(_pausedLoc.GetFormattedText());
		_resumeButton.GetNode<MegaLabel>("Label").SetTextAutoSize(_resumeLoc.GetFormattedText());
		_settingsButton.GetNode<MegaLabel>("Label").SetTextAutoSize(_settingsLoc.GetFormattedText());
		_compendiumButton.GetNode<MegaLabel>("Label").SetTextAutoSize(_compendiumLoc.GetFormattedText());
		_giveUpButton.GetNode<MegaLabel>("Label").SetTextAutoSize(_giveUpLoc.GetFormattedText());
		_disconnectButton.GetNode<MegaLabel>("Label").SetTextAutoSize(_disconnectLoc.GetFormattedText());
		_saveAndQuitButton.GetNode<MegaLabel>("Label").SetTextAutoSize(_saveAndQuitLoc.GetFormattedText());
	}

	public void Initialize(IRunState runState)
	{
		_runState = runState;
		if (!RunManager.Instance.IsInProgress || _runState.IsGameOver)
		{
			_giveUpButton.Disable();
		}
		else
		{
			_giveUpButton.Enable();
		}
		_compendiumButton.Visible = !NGame.IsReleaseGame() || SaveManager.Instance.IsCompendiumAvailable();
	}

	private void OnBackOrResumeButtonPressed(NButton _)
	{
		SfxCmd.Play("event:/sfx/ui/map/map_close");
		NCapstoneContainer.Instance.Close();
		NRun.Instance.GlobalUi.TopBar.Pause.ToggleAnimState();
	}

	private void OnSettingsButtonPressed(NButton _)
	{
		_stack.PushSubmenuType<NSettingsScreen>();
	}

	private void OnCompendiumButtonPressed(NButton _)
	{
		NCompendiumSubmenu submenuType = _stack.GetSubmenuType<NCompendiumSubmenu>();
		submenuType.Initialize(_runState);
		_stack.Push(submenuType);
	}

	private void OnGiveUpButtonPressed(NButton _)
	{
		NModalContainer.Instance.Add(NAbandonRunConfirmPopup.Create(null));
	}

	private void OnDisconnectButtonPressed(NButton _)
	{
		if (RunManager.Instance.NetService.IsConnected)
		{
			NModalContainer.Instance.Add(NDisconnectConfirmPopup.Create());
		}
		else
		{
			TaskHelper.RunSafely(NGame.Instance.ReturnToMainMenuAfterRun());
		}
	}

	private void OnSaveAndQuitButtonPressed(NButton _)
	{
		TaskHelper.RunSafely(CloseToMenu());
	}

	private async Task CloseToMenu()
	{
		_resumeButton.Disable();
		_settingsButton.Disable();
		_compendiumButton.Disable();
		_giveUpButton.Disable();
		_disconnectButton.Disable();
		_saveAndQuitButton.Disable();
		_backButton.Disable();
		NRunMusicController.Instance.StopMusic();
		await NGame.Instance.ReturnToMainMenu();
	}

	public override void OnSubmenuOpened()
	{
		base.OnSubmenuOpened();
		NHotkeyManager.Instance.AddBlockingScreen(this);
	}

	public override void OnSubmenuClosed()
	{
		_backButton.Disable();
		base.Visible = false;
		NHotkeyManager.Instance.RemoveBlockingScreen(this);
	}
}
