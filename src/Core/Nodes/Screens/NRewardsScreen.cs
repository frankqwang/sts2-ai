using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Ftue;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens;

public partial class NRewardsScreen : Control, IOverlayScreen, IScreenContext
{
	[Signal]
	public delegate void CompletedEventHandler();

	private const float _scrollLimitTop = 35f;

	private const int _scrollbarThreshold = 400;

	private IRunState _runState;

	private NProceedButton _proceedButton;

	private Control _rewardsContainer;

	private NScrollbar _scrollbar;

	private MegaLabel _headerLabel;

	private Control _rewardContainerMask;

	private Control _waitingForOtherPlayersOverlay;

	private Control _rewardsWindow;

	private Vector2 _targetDragPos;

	private bool _scrollbarPressed;

	private Tween? _fadeTween;

	private readonly List<Control> _rewardButtons = new List<Control>();

	private readonly List<Control> _skippedRewardButtons = new List<Control>();

	private Control? _lastRewardFocused;

	private bool _isTerminal;

	private readonly TaskCompletionSource _closedCompletionSource = new TaskCompletionSource();

	private static readonly LocString _waitingLoc = new LocString("gameplay_ui", "MULTIPLAYER_WAITING");

	private bool CanScroll => _rewardsContainer.Size.Y >= 400f;

	private float ScrollLimitBottom => 35f - _rewardsContainer.Size.Y + 400f;

	private static string ScenePath => SceneHelper.GetScenePath("screens/rewards_screen");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	public Task ClosedTask => _closedCompletionSource.Task;

	public bool IsComplete { get; private set; }

	public NetScreenType ScreenType => NetScreenType.Rewards;

	public bool UseSharedBackstop => true;

	public Control DefaultFocusedControl
	{
		get
		{
			if (_rewardButtons.Count == 0)
			{
				return _rewardsContainer;
			}
			return _lastRewardFocused ?? _rewardButtons[0];
		}
	}

	public Control FocusedControlFromTopBar
	{
		get
		{
			if (_rewardButtons.Count <= 0)
			{
				return _rewardsContainer;
			}
			return _rewardButtons[0];
		}
	}

	public static NRewardsScreen ShowScreen(bool isTerminal, IRunState runState)
	{
		NRewardsScreen nRewardsScreen = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NRewardsScreen>(PackedScene.GenEditState.Disabled);
		nRewardsScreen._isTerminal = isTerminal;
		nRewardsScreen._runState = runState;
		NOverlayStack.Instance.Push(nRewardsScreen);
		return nRewardsScreen;
	}

	public override void _Ready()
	{
		_proceedButton = GetNode<NProceedButton>("ProceedButton");
		_rewardContainerMask = GetNode<Control>("%RewardContainerMask");
		_rewardsContainer = GetNode<Control>("%RewardsContainer");
		_scrollbar = GetNode<NScrollbar>("%Scrollbar");
		_headerLabel = GetNode<MegaLabel>("%HeaderLabel");
		_waitingForOtherPlayersOverlay = GetNode<Control>("%WaitingForOtherPlayers");
		_waitingForOtherPlayersOverlay.GetNode<MegaLabel>("Label").SetTextAutoSize(_waitingLoc.GetRawText());
		_rewardsWindow = GetNode<Control>("Rewards");
		_rewardsWindow.Modulate = StsColors.transparentBlack;
		_proceedButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnProceedButtonPressed));
		_proceedButton.SetPulseState(isPulsing: false);
		TryEnableProceedButton();
		_proceedButton.UpdateText(NProceedButton.SkipLoc);
		NDebugAudioManager.Instance?.Play("victory.mp3");
		_scrollbar.Connect(NScrollbar.SignalName.MousePressed, Callable.From<InputEvent>(delegate
		{
			_scrollbarPressed = true;
		}));
		_scrollbar.Connect(NScrollbar.SignalName.MouseReleased, Callable.From<InputEvent>(delegate
		{
			_scrollbarPressed = false;
		}));
		_targetDragPos = new Vector2(_rewardsContainer.Position.X, 35f);
		if (_runState.CurrentRoom is CombatRoom { GoldProportion: <1f })
		{
			_headerLabel.SetTextAutoSize(new LocString("gameplay_ui", "COMBAT_REWARD_HEADER_MUGGED").GetFormattedText());
		}
		else
		{
			_headerLabel.SetTextAutoSize(new LocString("gameplay_ui", "COMBAT_REWARD_HEADER_LOOT").GetFormattedText());
		}
		GetViewport().Connect(Viewport.SignalName.GuiFocusChanged, Callable.From<Control>(ProcessGuiFocus));
		_rewardsContainer.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
		{
			DefaultFocusedControl.TryGrabFocus();
		}));
		ActiveScreenContext.Instance.Update();
	}

	public void SetRewards(IEnumerable<Reward> rewards)
	{
		foreach (Control rewardButton in _rewardButtons)
		{
			RemoveButton(rewardButton);
		}
		List<Reward> list = rewards.ToList();
		_rewardButtons.Clear();
		foreach (Reward item in list)
		{
			Control option;
			if (item is LinkedRewardSet linkedReward)
			{
				option = NLinkedRewardSet.Create(linkedReward, this);
				option.Connect(NLinkedRewardSet.SignalName.RewardClaimed, Callable.From<NLinkedRewardSet>(RewardCollectedFrom));
			}
			else
			{
				option = NRewardButton.Create(item, this);
				option.Connect(NRewardButton.SignalName.RewardClaimed, Callable.From<NRewardButton>(RewardCollectedFrom));
				option.Connect(NRewardButton.SignalName.RewardSkipped, Callable.From<NRewardButton>(RewardSkippedFrom));
				option.Connect(Control.SignalName.FocusEntered, Callable.From(() => _lastRewardFocused = option));
			}
			item.MarkContentAsSeen();
			_rewardButtons.Add(option);
			_rewardsContainer.AddChildSafely(option);
		}
		UpdateScreenState();
		if (list.Count == 0)
		{
			TryEnableProceedButton();
		}
		if (_rewardsContainer.HasFocus())
		{
			DefaultFocusedControl.TryGrabFocus();
		}
		TaskHelper.RunSafely(RelicFtueCheck());
	}

	private async Task RelicFtueCheck()
	{
		if (SaveManager.Instance.SeenFtue("obtain_relic_ftue"))
		{
			return;
		}
		await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		foreach (NRewardButton item in _rewardButtons.OfType<NRewardButton>())
		{
			if (item.Reward is RelicReward)
			{
				NModalContainer.Instance.Add(NRelicRewardFtue.Create(item));
				SaveManager.Instance.MarkFtueAsComplete("obtain_relic_ftue");
				break;
			}
		}
	}

	public void RewardCollectedFrom(Control button)
	{
		int a = _rewardButtons.IndexOf(button);
		RemoveButton(button);
		_lastRewardFocused = ((_rewardButtons.Count > 0) ? _rewardButtons[Mathf.Min(a, _rewardButtons.Count - 1)] : null);
		UpdateScreenState();
		if (_rewardButtons.Count > 0 || _isTerminal)
		{
			TryEnableProceedButton();
			if (!_rewardButtons.Except(_skippedRewardButtons).Any())
			{
				_proceedButton.SetPulseState(isPulsing: true);
			}
		}
	}

	public void RewardSkippedFrom(Control button)
	{
		_skippedRewardButtons.Add(button);
		if (!_rewardButtons.Except(_skippedRewardButtons).Any())
		{
			_proceedButton.SetPulseState(isPulsing: true);
		}
	}

	private void UpdateScreenState()
	{
		if (_rewardButtons.Count == 0)
		{
			if (_isTerminal)
			{
				_fadeTween?.Kill();
				_fadeTween = CreateTween().SetParallel();
				_fadeTween.TweenProperty(GetNode<Control>("Rewards"), "modulate:a", 0f, 0.25);
				NOverlayStack.Instance.HideBackstop();
				_proceedButton.UpdateText(NProceedButton.ProceedLoc);
				_proceedButton.SetPulseState(isPulsing: true);
				_rewardsContainer.FocusMode = FocusModeEnum.None;
				IsComplete = true;
				EmitSignal(SignalName.Completed);
			}
			else
			{
				NOverlayStack.Instance.Remove(this);
			}
		}
		_rewardsContainer.ResetSize();
		_scrollbar.Visible = CanScroll;
		_scrollbar.MouseFilter = (MouseFilterEnum)(CanScroll ? 0 : 2);
		if (!CanScroll)
		{
			_targetDragPos.Y = 35f;
		}
		for (int i = 0; i < _rewardButtons.Count; i++)
		{
			_rewardButtons[i].FocusNeighborLeft = _rewardButtons[i].GetPath();
			_rewardButtons[i].FocusNeighborRight = _rewardButtons[i].GetPath();
			_rewardButtons[i].FocusNeighborTop = ((i > 0) ? _rewardButtons[i - 1].GetPath() : _rewardButtons[i].GetPath());
			_rewardButtons[i].FocusNeighborBottom = ((i < _rewardButtons.Count - 1) ? _rewardButtons[i + 1].GetPath() : _rewardButtons[i].GetPath());
		}
	}

	private void RemoveButton(Control button)
	{
		button.GetParent().RemoveChildSafely(button);
		button.QueueFreeSafely();
		int a = _rewardButtons.IndexOf(button);
		_rewardButtons.Remove(button);
		if (_rewardButtons.Count > 0)
		{
			a = Mathf.Min(a, _rewardButtons.Count - 1);
			_rewardButtons[a].TryGrabFocus();
		}
		else if (_rewardButtons.Contains(GetViewport().GuiGetFocusOwner()))
		{
			ActiveScreenContext.Instance.Update();
		}
	}

	private void OnProceedButtonPressed(NButton _)
	{
		if (RunManager.Instance.debugAfterCombatRewardsOverride != null && _isTerminal)
		{
			RunManager.Instance.debugAfterCombatRewardsOverride?.Invoke();
		}
		else if (_isTerminal && (_runState.CurrentRoom.RoomType == RoomType.Boss || _runState.CurrentRoom.IsVictoryRoom))
		{
			if (_runState.Map.SecondBossMapPoint != null && _runState.CurrentMapCoord == _runState.Map.BossMapPoint.coord)
			{
				TaskHelper.RunSafely(RunManager.Instance.ProceedFromTerminalRewardsScreen());
				return;
			}
			_proceedButton.Disable();
			if (RunManager.Instance.ActChangeSynchronizer.IsWaitingForOtherPlayers())
			{
				_waitingForOtherPlayersOverlay.Visible = true;
			}
			RunManager.Instance.ActChangeSynchronizer.SetLocalPlayerReady();
		}
		else if (_isTerminal)
		{
			if (_proceedButton.IsSkip)
			{
				if (TestMode.IsOn || SaveManager.Instance.SeenFtue("combat_reward_ftue"))
				{
					TaskHelper.RunSafely(RunManager.Instance.ProceedFromTerminalRewardsScreen());
				}
				else
				{
					TaskHelper.RunSafely(RewardFtueCheck());
				}
				return;
			}
			if (_runState.ActFloor > 4)
			{
				SaveManager.Instance.MarkFtueAsComplete("combat_reward_ftue");
			}
			TaskHelper.RunSafely(RunManager.Instance.ProceedFromTerminalRewardsScreen());
		}
		else
		{
			NOverlayStack.Instance.Remove(this);
		}
	}

	public void AfterOverlayOpened()
	{
	}

	public void AfterOverlayClosed()
	{
		if (RunManager.Instance.IsInProgress && !RunManager.Instance.IsCleaningUp)
		{
			foreach (NRewardButton item in _rewardsContainer.GetChildren().OfType<NRewardButton>())
			{
				item.Reward?.OnSkipped();
			}
			foreach (NLinkedRewardSet item2 in _rewardsContainer.GetChildren().OfType<NLinkedRewardSet>())
			{
				item2.LinkedRewardSet.OnSkipped();
			}
			_closedCompletionSource.SetResult();
		}
		_proceedButton.Disable();
		this.QueueFreeSafely();
	}

	private void TryEnableProceedButton()
	{
		if (Hook.ShouldProceedToNextMapPoint(_runState) && !_proceedButton.IsEnabled)
		{
			if (_isTerminal && _rewardButtons.Count == 0)
			{
				NOverlayStack.Instance.HideBackstop();
			}
			_proceedButton.Enable();
		}
	}

	public void AfterOverlayShown()
	{
		TryEnableProceedButton();
		if (!IsComplete)
		{
			_fadeTween?.FastForwardToCompletion();
			_fadeTween = CreateTween().SetParallel();
			_fadeTween.TweenProperty(_rewardsWindow, "modulate", Colors.White, 0.5);
			_fadeTween.TweenProperty(_rewardsWindow, "position:y", _rewardsWindow.Position.Y, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back)
				.From(_rewardsWindow.Position.Y + 100f);
		}
	}

	public void AfterOverlayHidden()
	{
		_proceedButton.Disable();
		if (!IsComplete)
		{
			_fadeTween?.FastForwardToCompletion();
			_fadeTween = CreateTween();
			_fadeTween.TweenProperty(_rewardsWindow, "modulate:a", 0, 0.25);
		}
	}

	public override void _GuiInput(InputEvent inputEvent)
	{
		if (IsVisibleInTree() && CanScroll)
		{
			ProcessScrollEvent(inputEvent);
		}
	}

	private void ProcessScrollEvent(InputEvent inputEvent)
	{
		_targetDragPos += new Vector2(0f, ScrollHelper.GetDragForScrollEvent(inputEvent));
	}

	private void ProcessGuiFocus(Control focusedControl)
	{
		if (IsVisibleInTree() && CanScroll && NControllerManager.Instance.IsUsingController && _rewardButtons.Contains(focusedControl))
		{
			float value = 0f - focusedControl.Position.Y + _rewardContainerMask.Size.Y * 0.5f;
			value = Mathf.Clamp(value, ScrollLimitBottom, 35f);
			_targetDragPos = new Vector2(_targetDragPos.X, value);
		}
	}

	public override void _Process(double delta)
	{
		if (IsVisibleInTree())
		{
			UpdateScrollPosition(delta);
		}
	}

	private void UpdateScrollPosition(double delta)
	{
		if (!_rewardsContainer.Position.IsEqualApprox(_targetDragPos))
		{
			_rewardsContainer.Position = _rewardsContainer.Position.Lerp(_targetDragPos, Mathf.Clamp((float)delta * 15f, 0f, 1f));
			if (_rewardsContainer.Position.DistanceTo(_targetDragPos) < 0.5f)
			{
				_rewardsContainer.Position = _targetDragPos;
			}
			if (!_scrollbarPressed && CanScroll)
			{
				_scrollbar.SetValueWithoutAnimation(Mathf.Clamp(_rewardsContainer.Position.Y / ScrollLimitBottom, 0f, 1f) * 100f);
			}
		}
		if (_scrollbarPressed)
		{
			_targetDragPos.Y = Mathf.Lerp(35f, ScrollLimitBottom, (float)_scrollbar.Value * 0.01f);
		}
		if (_targetDragPos.Y < Mathf.Min(ScrollLimitBottom, 0f))
		{
			_targetDragPos.Y = Mathf.Lerp(_targetDragPos.Y, ScrollLimitBottom, (float)delta * 12f);
		}
		else if (_targetDragPos.Y > Mathf.Max(ScrollLimitBottom, 0f))
		{
			_targetDragPos.Y = Mathf.Lerp(_targetDragPos.Y, 35f, (float)delta * 12f);
		}
	}

	private async Task RewardFtueCheck()
	{
		_proceedButton.Hide();
		NCombatRewardFtue nCombatRewardFtue = NCombatRewardFtue.Create(_rewardsContainer);
		NModalContainer.Instance.Add(nCombatRewardFtue);
		SaveManager.Instance.MarkFtueAsComplete("combat_reward_ftue");
		await nCombatRewardFtue.WaitForPlayerToConfirm();
		_proceedButton.Show();
	}

	public void HideWaitingForPlayersScreen()
	{
		_waitingForOtherPlayersOverlay.Visible = false;
	}
}
