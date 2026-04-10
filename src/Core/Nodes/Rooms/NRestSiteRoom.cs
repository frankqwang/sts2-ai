using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Debug;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.Audio;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Ftue;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.RestSite;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Nodes.Vfx;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Rooms;

public partial class NRestSiteRoom : Control, IScreenContext, IRoomWithProceedButton
{
	public readonly List<NRestSiteCharacter> characterAnims = new List<NRestSiteCharacter>();

	private const float _lowDescriptionYPos = 885f;

	private const string _scenePath = "res://scenes/rooms/rest_site_room.tscn";

	private static bool _isDebugUiVisible;

	private RestSiteRoom _room;

	private IRunState _runState;

	private Control _choicesContainer;

	private Control _choicesScreen;

	private NProceedButton _proceedButton;

	private Control _restSiteLighting;

	private readonly List<Control> _characterContainers = new List<Control>();

	private readonly CancellationTokenSource _cts = new CancellationTokenSource();

	private Tween? _descriptionTween;

	private Tween? _descriptionPositionTween;

	private Tween? _choicesTween;

	private float _originalDescriptionYPos;

	private Control? _lastFocused;

	public static NRestSiteRoom? Instance => NRun.Instance?.RestSiteRoom;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>("res://scenes/rooms/rest_site_room.tscn");

	public NProceedButton ProceedButton => _proceedButton;

	private MegaLabel Header { get; set; }

	private MegaRichTextLabel Description { get; set; }

	private Control BgContainer { get; set; }

	public IReadOnlyList<RestSiteOption> Options => _room.Options;

	public List<NRestSiteCharacter> Characters { get; } = new List<NRestSiteCharacter>();

	public Control? DefaultFocusedControl
	{
		get
		{
			if (_lastFocused != null)
			{
				return _lastFocused;
			}
			if (_choicesContainer.GetChildCount() <= 0)
			{
				return null;
			}
			return _choicesContainer.GetChild<NRestSiteButton>(0);
		}
	}

	public static NRestSiteRoom? Create(RestSiteRoom room, IRunState runState)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NRestSiteRoom nRestSiteRoom = PreloadManager.Cache.GetScene("res://scenes/rooms/rest_site_room.tscn").Instantiate<NRestSiteRoom>(PackedScene.GenEditState.Disabled);
		nRestSiteRoom._room = room;
		nRestSiteRoom._runState = runState;
		return nRestSiteRoom;
	}

	public override void _Ready()
	{
		Header = GetNode<MegaLabel>("%Header");
		Description = GetNode<MegaRichTextLabel>("%Description");
		Description.Modulate = Colors.Transparent;
		_choicesContainer = GetNode<Control>("%ChoicesContainer");
		_choicesScreen = GetNode<Control>("%ChoicesScreen");
		_proceedButton = GetNode<NProceedButton>("%ProceedButton");
		BgContainer = GetNode<Control>("BgContainer");
		_characterContainers.Add(GetNode<Control>("BgContainer/Character_1"));
		_characterContainers.Add(GetNode<Control>("BgContainer/Character_2"));
		_characterContainers.Add(GetNode<Control>("BgContainer/Character_3"));
		_characterContainers.Add(GetNode<Control>("BgContainer/Character_4"));
		Control control = _runState.Act.CreateRestSiteBackground();
		BgContainer.AddChildSafely(control);
		BgContainer.MoveChild(control, 0);
		_restSiteLighting = control.GetNode<Control>("%RestSiteLighting");
		Header.SetTextAutoSize(new LocString("rest_site_ui", "PROMPT").GetFormattedText());
		_proceedButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnProceedButtonReleased));
		_proceedButton.UpdateText(NProceedButton.ProceedLoc);
		if (_isDebugUiVisible)
		{
			_choicesScreen.Modulate = Colors.Transparent;
		}
		NGame.Instance.SetScreenShakeTarget(this);
		_proceedButton.Disable();
		for (int i = 0; i < _runState.Players.Count; i++)
		{
			NRestSiteCharacter nRestSiteCharacter = NRestSiteCharacter.Create(_runState.Players[i], i);
			characterAnims.Add(nRestSiteCharacter);
			_characterContainers[i].AddChildSafely(nRestSiteCharacter);
			nRestSiteCharacter.Position = Vector2.Zero;
			if (i % 2 == 1)
			{
				nRestSiteCharacter.FlipX();
			}
			Characters.Add(nRestSiteCharacter);
		}
		_originalDescriptionYPos = Description.Position.Y;
		UpdateRestSiteOptions();
		TaskHelper.RunSafely(ShowFtueIfNeeded());
		RunManager.Instance.RestSiteSynchronizer.PlayerHoverChanged += OnPlayerChangedHoveredRestSiteOption;
		RunManager.Instance.RestSiteSynchronizer.BeforePlayerOptionChosen += OnBeforePlayerSelectedRestSiteOption;
		RunManager.Instance.RestSiteSynchronizer.AfterPlayerOptionChosen += OnAfterPlayerSelectedRestSiteOption;
	}

	public override void _EnterTree()
	{
		ActiveScreenContext.Instance.Updated += OnActiveScreenUpdated;
	}

	public override void _ExitTree()
	{
		_cts.Cancel();
		_cts.Dispose();
		_descriptionTween?.Kill();
		_descriptionPositionTween?.Kill();
		_choicesTween?.Kill();
		RunManager.Instance.RestSiteSynchronizer.PlayerHoverChanged -= OnPlayerChangedHoveredRestSiteOption;
		RunManager.Instance.RestSiteSynchronizer.BeforePlayerOptionChosen -= OnBeforePlayerSelectedRestSiteOption;
		RunManager.Instance.RestSiteSynchronizer.AfterPlayerOptionChosen -= OnAfterPlayerSelectedRestSiteOption;
		ActiveScreenContext.Instance.Updated -= OnActiveScreenUpdated;
	}

	public void AfterSelectingOption(RestSiteOption option)
	{
		TaskHelper.RunSafely(AfterSelectingOptionAsync(option));
	}

	private async Task ShowFtueIfNeeded()
	{
		if (!SaveManager.Instance.SeenFtue("rest_site_ftue"))
		{
			_choicesContainer.Visible = false;
			Header.Visible = false;
			await Cmd.Wait(0.5f, _cts.Token);
			_choicesContainer.Visible = true;
			Header.Visible = true;
			Control choicesContainer = _choicesContainer;
			Color modulate = _choicesContainer.Modulate;
			modulate.A = 0f;
			choicesContainer.Modulate = modulate;
			MegaLabel header = Header;
			modulate = Header.Modulate;
			modulate.A = 0f;
			header.Modulate = modulate;
			Tween tween = CreateTween();
			tween.TweenProperty(_choicesContainer, "modulate:a", 1f, 0.5);
			tween.TweenProperty(Header, "modulate:a", 1f, 0.5);
			NModalContainer.Instance.Add(NRestSiteFtue.Create(_choicesContainer));
			SaveManager.Instance.MarkFtueAsComplete("rest_site_ftue");
		}
	}

	public void DisableOptions()
	{
		foreach (NRestSiteButton item in _choicesContainer.GetChildren().OfType<NRestSiteButton>())
		{
			item.Disable();
		}
	}

	public void EnableOptions()
	{
		foreach (NRestSiteButton item in _choicesContainer.GetChildren().OfType<NRestSiteButton>())
		{
			item.Enable();
		}
		ActiveScreenContext.Instance.Update();
	}

	public void AnimateDescriptionDown()
	{
		_descriptionPositionTween?.Kill();
		_descriptionPositionTween = CreateTween();
		_descriptionPositionTween.TweenProperty(Description, "position:y", 885f, 0.800000011920929).SetTrans(Tween.TransitionType.Expo).SetEase(Tween.EaseType.Out);
	}

	public void AnimateDescriptionUp()
	{
		_descriptionPositionTween?.Kill();
		_descriptionPositionTween = CreateTween();
		_descriptionPositionTween.TweenProperty(Description, "position:y", _originalDescriptionYPos, 0.800000011920929).SetTrans(Tween.TransitionType.Expo).SetEase(Tween.EaseType.Out);
	}

	private void UpdateRestSiteOptions()
	{
		if (!this.IsValid() || !IsInsideTree())
		{
			return;
		}
		foreach (Node child in _choicesContainer.GetChildren())
		{
			child.QueueFreeSafely();
		}
		foreach (RestSiteOption option in Options)
		{
			NRestSiteButton nRestSiteButton = NRestSiteButton.Create(option);
			_choicesContainer.AddChildSafely(nRestSiteButton);
			nRestSiteButton.Connect(NClickableControl.SignalName.Focused, Callable.From<NRestSiteButton>(RestSiteButtonHovered));
			nRestSiteButton.Connect(NClickableControl.SignalName.Unfocused, Callable.From<NRestSiteButton>(RestSiteButtonUnhovered));
		}
	}

	private void RestSiteButtonHovered(NRestSiteButton button)
	{
		RunManager.Instance.RestSiteSynchronizer.LocalOptionHovered(button.Option);
		_lastFocused = button;
	}

	private void RestSiteButtonUnhovered(NRestSiteButton button)
	{
		RunManager.Instance.RestSiteSynchronizer.LocalOptionHovered(null);
	}

	private void OnPlayerChangedHoveredRestSiteOption(ulong playerId)
	{
		if (_runState.Players.Count > 1)
		{
			NRestSiteCharacter nRestSiteCharacter = Characters.First((NRestSiteCharacter c) => c.Player.NetId == playerId);
			int? hoveredOptionIndex = RunManager.Instance.RestSiteSynchronizer.GetHoveredOptionIndex(playerId);
			RestSiteOption option = ((!hoveredOptionIndex.HasValue) ? null : RunManager.Instance.RestSiteSynchronizer.GetOptionsForPlayer(playerId)[hoveredOptionIndex.Value]);
			nRestSiteCharacter.ShowHoveredRestSiteOption(option);
		}
	}

	private void OnBeforePlayerSelectedRestSiteOption(RestSiteOption option, ulong playerId)
	{
		if (_runState.Players.Count > 1)
		{
			NRestSiteCharacter nRestSiteCharacter = Characters.First((NRestSiteCharacter c) => c.Player.NetId == playerId);
			nRestSiteCharacter.SetSelectingRestSiteOption(option);
		}
	}

	private void OnAfterPlayerSelectedRestSiteOption(RestSiteOption option, bool success, ulong playerId)
	{
		if (_runState.Players.Count <= 1)
		{
			return;
		}
		NRestSiteCharacter nRestSiteCharacter = Characters.First((NRestSiteCharacter c) => c.Player.NetId == playerId);
		nRestSiteCharacter.SetSelectingRestSiteOption(null);
		if (success)
		{
			nRestSiteCharacter.ShowSelectedRestSiteOption(option);
			if (!LocalContext.IsMe(nRestSiteCharacter.Player))
			{
				TaskHelper.RunSafely(option.DoRemotePostSelectVfx());
			}
		}
	}

	public NRestSiteButton? GetButtonForOption(RestSiteOption option)
	{
		foreach (NRestSiteButton item in _choicesContainer.GetChildren().OfType<NRestSiteButton>())
		{
			if (item.Option == option)
			{
				return item;
			}
		}
		return null;
	}

	public NRestSiteCharacter? GetCharacterForPlayer(Player player)
	{
		foreach (NRestSiteCharacter characterAnim in characterAnims)
		{
			if (characterAnim.Player == player)
			{
				return characterAnim;
			}
		}
		return null;
	}

	private async Task AfterSelectingOptionAsync(RestSiteOption option)
	{
		Task task = HideChoices(_cts.Token);
		Task task2 = option.DoLocalPostSelectVfx(_cts.Token);
		ExtinguishFireIfAble();
		global::_003C_003Ey__InlineArray2<Task> buffer = default(global::_003C_003Ey__InlineArray2<Task>);
		buffer[0] = task;
		buffer[1] = task2;
		await Task.WhenAll(buffer);
		UpdateRestSiteOptions();
		ShowProceedButton();
		if (Options.Count > 0)
		{
			await ShowChoices(_cts.Token);
			ActiveScreenContext.Instance.Update();
		}
	}

	private void ShowProceedButton()
	{
		if (!_proceedButton.IsEnabled)
		{
			_proceedButton.Enable();
			NMapScreen.Instance.SetTravelEnabled(enabled: true);
		}
	}

	private void OnProceedButtonReleased(NButton _)
	{
		NMapScreen.Instance.Open();
	}

	public void SetText(string formattedText)
	{
		_descriptionTween?.Kill();
		Description.Modulate = Colors.White;
		Description.Text = formattedText;
	}

	public void FadeOutOptionDescription()
	{
		_descriptionTween?.Kill();
		_descriptionTween = CreateTween();
		_descriptionTween.TweenProperty(Description, "modulate:a", 0f, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
			.From(1f);
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (inputEvent.IsActionReleased(DebugHotkey.hideRestSite))
		{
			_isDebugUiVisible = !_isDebugUiVisible;
			_choicesScreen.Modulate = (_isDebugUiVisible ? Colors.Transparent : Colors.White);
			NGame.Instance.AddChildSafely(NFullscreenTextVfx.Create(_isDebugUiVisible ? "Hide RestSite UI" : "Show RestSite UI"));
		}
	}

	private void ExtinguishFireIfAble()
	{
		if (RunManager.Instance.RestSiteSynchronizer.GetLocalOptions().Count > 0 || !_restSiteLighting.Visible)
		{
			return;
		}
		foreach (NRestSiteCharacter characterAnim in characterAnims)
		{
			characterAnim.HideFlameGlow();
		}
		foreach (Control characterContainer in _characterContainers)
		{
			characterContainer.Modulate = Colors.DarkGray;
		}
		_restSiteLighting.Visible = false;
		NRunMusicController.Instance?.TriggerCampfireGoingOut();
	}

	private async Task ShowChoices(CancellationToken ct)
	{
		_choicesTween?.Kill();
		_choicesTween = CreateTween();
		_choicesTween.TweenProperty(_choicesScreen, "modulate:a", 1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		await _choicesTween.AwaitFinished(ct);
	}

	private async Task HideChoices(CancellationToken ct)
	{
		foreach (NButton item in _choicesContainer.GetChildren().OfType<NButton>())
		{
			item.Disable();
		}
		_choicesTween?.Kill();
		_choicesTween = CreateTween();
		_choicesTween.TweenProperty(_choicesScreen, "modulate:a", 0f, 0.5);
		_lastFocused = null;
		await _choicesTween.AwaitFinished(ct);
	}

	private void OnActiveScreenUpdated()
	{
		this.UpdateControllerNavEnabled();
		if (ActiveScreenContext.Instance.IsCurrent(this) && Options.Count == 0)
		{
			ShowProceedButton();
		}
		else if (!_proceedButton.IsEnabled)
		{
			_proceedButton.Disable();
		}
	}
}
