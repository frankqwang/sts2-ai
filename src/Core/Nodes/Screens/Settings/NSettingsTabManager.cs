using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NSettingsTabManager : Control
{
	[Signal]
	public delegate void TabChangedEventHandler();

	private const float _scrollPaddingTop = 20f;

	private const float _scrollPaddingBottom = 30f;

	private static readonly StringName _tabLeftHotkey = MegaInput.viewDeckAndTabLeft;

	private static readonly StringName _tabRightHotkey = MegaInput.viewExhaustPileAndTabRight;

	private NSettingsTab? _currentTab;

	private NScrollableContainer _scrollContainer;

	private readonly Dictionary<NSettingsTab, NSettingsPanel> _tabs = new Dictionary<NSettingsTab, NSettingsPanel>();

	private TextureRect _leftTriggerIcon;

	private TextureRect _rightTriggerIcon;

	private Tween? _scrollbarTween;

	private NSettingsPanel CurrentlyDisplayedPanel => _tabs[_currentTab];

	public Control? DefaultFocusedControl
	{
		get
		{
			if (_currentTab == null)
			{
				return null;
			}
			return _tabs[_currentTab].DefaultFocusedControl;
		}
	}

	public override void _Ready()
	{
		_leftTriggerIcon = GetNode<TextureRect>("LeftTriggerIcon");
		_rightTriggerIcon = GetNode<TextureRect>("RightTriggerIcon");
		_scrollContainer = GetNode<NScrollableContainer>("%ScrollContainer");
		_scrollContainer.DisableScrollingIfContentFits();
		NSettingsTab node = GetNode<NSettingsTab>("General");
		node.SetLabel(new LocString("settings_ui", "TAB_GENERAL").GetFormattedText());
		_tabs.Add(node, GetNode<NSettingsPanel>("%GeneralSettings"));
		node = GetNode<NSettingsTab>("Graphics");
		node.SetLabel(new LocString("settings_ui", "TAB_GRAPHICS").GetFormattedText());
		_tabs.Add(node, GetNode<NSettingsPanel>("%GraphicsSettings"));
		node = GetNode<NSettingsTab>("Sound");
		node.SetLabel(new LocString("settings_ui", "TAB_SOUND").GetFormattedText());
		_tabs.Add(node, GetNode<NSettingsPanel>("%SoundSettings"));
		node = GetNode<NSettingsTab>("Input");
		node.SetLabel(new LocString("settings_ui", "TAB_INPUT").GetFormattedText());
		_tabs.Add(node, GetNode<NSettingsPanel>("%InputSettings"));
		foreach (NSettingsTab tab in _tabs.Keys)
		{
			tab.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(delegate
			{
				SwitchTabTo(tab);
			}));
		}
		NControllerManager.Instance.Connect(NControllerManager.SignalName.MouseDetected, Callable.From(UpdateControllerButton));
		NControllerManager.Instance.Connect(NControllerManager.SignalName.ControllerDetected, Callable.From(UpdateControllerButton));
		NInputManager.Instance.Connect(NInputManager.SignalName.InputRebound, Callable.From(UpdateControllerButton));
		UpdateControllerButton();
	}

	public void ResetTabs()
	{
		_tabs.First().Key.Select();
		SwitchTabTo(_tabs.First().Key);
	}

	public void Enable()
	{
		NHotkeyManager.Instance.PushHotkeyPressedBinding(_tabLeftHotkey, TabLeft);
		NHotkeyManager.Instance.PushHotkeyPressedBinding(_tabRightHotkey, TabRight);
	}

	public void Disable()
	{
		NHotkeyManager.Instance.RemoveHotkeyPressedBinding(_tabLeftHotkey, TabLeft);
		NHotkeyManager.Instance.RemoveHotkeyPressedBinding(_tabRightHotkey, TabRight);
	}

	private void TabLeft()
	{
		List<NSettingsTab> list = _tabs.Keys.ToList();
		int num = list.IndexOf(_currentTab) - 1;
		if (num >= 0)
		{
			SwitchTabTo(list[num]);
		}
	}

	private void TabRight()
	{
		List<NSettingsTab> list = _tabs.Keys.ToList();
		int num = Math.Min(list.Count - 1, list.IndexOf(_currentTab) + 1);
		if (num < list.Count)
		{
			SwitchTabTo(list[num]);
		}
	}

	private void SwitchTabTo(NSettingsTab selectedTab)
	{
		if (selectedTab != _currentTab)
		{
			foreach (NSettingsTab key in _tabs.Keys)
			{
				key.Deselect();
				_tabs[key].Visible = false;
			}
			selectedTab.Select();
			_tabs[selectedTab].Visible = true;
			_currentTab = selectedTab;
			_scrollContainer.SetContent(CurrentlyDisplayedPanel, 20f, 30f);
			_scrollContainer.InstantlyScrollToTop();
			_scrollbarTween?.Kill();
			_scrollbarTween = CreateTween().SetParallel();
			_scrollbarTween.TweenProperty(_scrollContainer.Scrollbar, "modulate", Colors.White, 0.5).From(StsColors.transparentBlack).SetEase(Tween.EaseType.Out)
				.SetTrans(Tween.TransitionType.Cubic);
		}
		ActiveScreenContext.Instance.Update();
	}

	private void UpdateControllerButton()
	{
		_leftTriggerIcon.Visible = NControllerManager.Instance.IsUsingController;
		_rightTriggerIcon.Visible = NControllerManager.Instance.IsUsingController;
		_leftTriggerIcon.Texture = NInputManager.Instance.GetHotkeyIcon(MegaInput.viewDeckAndTabLeft);
		_rightTriggerIcon.Texture = NInputManager.Instance.GetHotkeyIcon(MegaInput.viewExhaustPileAndTabRight);
	}
}
