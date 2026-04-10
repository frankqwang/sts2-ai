using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Debug;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.Settings;

namespace MegaCrit.Sts2.Core.Nodes.Screens.StatsScreen;

public partial class NStatsTabManager : Control
{
	private static readonly StringName _tabLeftHotkey = MegaInput.viewDeckAndTabLeft;

	private static readonly StringName _tabRightHotkey = MegaInput.viewExhaustPileAndTabRight;

	private Control _leftTriggerIcon;

	private Control _rightTriggerIcon;

	private Control _tabContainer;

	private List<NSettingsTab> _tabs;

	private NSettingsTab? _currentTab;

	public override void _Ready()
	{
		_leftTriggerIcon = GetNode<Control>("LeftTriggerIcon");
		_rightTriggerIcon = GetNode<Control>("RightTriggerIcon");
		_tabContainer = GetNode<Control>("TabContainer");
		_tabs = _tabContainer.GetChildren().OfType<NSettingsTab>().ToList();
		NControllerManager.Instance.Connect(NControllerManager.SignalName.MouseDetected, Callable.From(UpdateControllerButton));
		NControllerManager.Instance.Connect(NControllerManager.SignalName.ControllerDetected, Callable.From(UpdateControllerButton));
		NInputManager.Instance.Connect(NInputManager.SignalName.InputRebound, Callable.From(UpdateControllerButton));
		foreach (NSettingsTab nSettingsTab in _tabs)
		{
			nSettingsTab.Connect(NClickableControl.SignalName.Released, Callable.From<NClickableControl>(delegate
			{
				SwitchToTab(nSettingsTab);
			}));
		}
		UpdateControllerButton();
	}

	public void ResetTabs()
	{
		SwitchToTab(_tabContainer.GetChild<NSettingsTab>(0));
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (!IsVisibleInTree() || NDevConsole.Instance.Visible)
		{
			return;
		}
		Control control = GetViewport().GuiGetFocusOwner();
		if ((control is TextEdit || control is LineEdit) ? true : false)
		{
			return;
		}
		if (inputEvent.IsActionPressed(_tabLeftHotkey))
		{
			int num = _tabs.IndexOf(_currentTab) - 1;
			if (num >= 0)
			{
				_tabs[num].ForceTabPressed();
				SwitchToTab(_tabs[num]);
			}
		}
		if (inputEvent.IsActionPressed(_tabRightHotkey))
		{
			int num2 = Math.Min(_tabs.Count - 1, _tabs.IndexOf(_currentTab) + 1);
			if (num2 < _tabs.Count)
			{
				_tabs[num2].ForceTabPressed();
			}
		}
	}

	private void SwitchToTab(NSettingsTab tab)
	{
		_currentTab = tab;
		foreach (NSettingsTab tab2 in _tabs)
		{
			if (tab2 != _currentTab)
			{
				tab2.Deselect();
			}
			else
			{
				tab2.Select();
			}
		}
	}

	private void UpdateControllerButton()
	{
		_leftTriggerIcon.Visible = NControllerManager.Instance.IsUsingController;
		_rightTriggerIcon.Visible = NControllerManager.Instance.IsUsingController;
	}
}
