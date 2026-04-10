using System;
using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Screens.Capstones;
using MegaCrit.Sts2.Core.Nodes.Screens.FeedbackScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Nodes.Screens.PauseMenu;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Nodes.Screens.Settings;

namespace MegaCrit.Sts2.Core.Nodes.Screens;

public partial class NCapstoneSubmenuStack : Control, ICapstoneScreen, IScreenContext
{
	private static string ScenePath => SceneHelper.GetScenePath("screens/capstone_submenu_stack");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	public CapstoneSubmenuType Type { get; private set; }

	public NRunSubmenuStack Stack { get; private set; }

	public NetScreenType ScreenType => GetCapstoneSubmenuType();

	public bool UseSharedBackstop => true;

	public Control? DefaultFocusedControl => Stack.Peek()?.DefaultFocusedControl;

	public NSubmenu ShowScreen(CapstoneSubmenuType type)
	{
		while (Stack.Peek() != null)
		{
			Stack.Pop();
		}
		Type type2 = type switch
		{
			CapstoneSubmenuType.Compendium => typeof(NCompendiumSubmenu), 
			CapstoneSubmenuType.Feedback => typeof(NSendFeedbackScreen), 
			CapstoneSubmenuType.PauseMenu => typeof(NPauseMenu), 
			CapstoneSubmenuType.Settings => typeof(NSettingsScreen), 
			_ => throw new ArgumentOutOfRangeException("type", type, null), 
		};
		NSubmenu result = Stack.PushSubmenuType(type2);
		Type = type;
		NCapstoneContainer.Instance.Open(this);
		return result;
	}

	private NetScreenType GetCapstoneSubmenuType()
	{
		return Type switch
		{
			CapstoneSubmenuType.Compendium => NetScreenType.Compendium, 
			CapstoneSubmenuType.Feedback => NetScreenType.Feedback, 
			CapstoneSubmenuType.PauseMenu => NetScreenType.PauseMenu, 
			CapstoneSubmenuType.Settings => NetScreenType.Settings, 
			_ => throw new ArgumentOutOfRangeException(), 
		};
	}

	public override void _Ready()
	{
		Stack = GetNode<NRunSubmenuStack>("%Submenus");
		Stack.Connect(NSubmenuStack.SignalName.StackModified, Callable.From(OnSubmenuStackChanged));
	}

	private void OnSubmenuStackChanged()
	{
		if (Stack.Peek() == null && NCapstoneContainer.Instance.CurrentCapstoneScreen == this)
		{
			NCapstoneContainer.Instance.Close();
		}
	}

	public void AfterCapstoneOpened()
	{
		NGlobalUi globalUi = NRun.Instance.GlobalUi;
		globalUi.TopBar.AnimHide();
		globalUi.RelicInventory.AnimHide();
		globalUi.MultiplayerPlayerContainer.AnimHide();
		SfxCmd.Play("event:/sfx/ui/pause_open");
		globalUi.MoveChild(globalUi.AboveTopBarVfxContainer, globalUi.CapstoneContainer.GetIndex());
		globalUi.MoveChild(globalUi.CardPreviewContainer, globalUi.CapstoneContainer.GetIndex());
		globalUi.MoveChild(globalUi.MessyCardPreviewContainer, globalUi.CapstoneContainer.GetIndex());
		base.Visible = true;
	}

	public void AfterCapstoneClosed()
	{
		while (Stack.Peek() != null)
		{
			Stack.Pop();
		}
		SfxCmd.Play("event:/sfx/ui/pause_close");
		NGlobalUi globalUi = NRun.Instance.GlobalUi;
		globalUi.TopBar.AnimShow();
		globalUi.RelicInventory.AnimShow();
		globalUi.MultiplayerPlayerContainer.AnimShow();
		globalUi.MoveChild(globalUi.AboveTopBarVfxContainer, globalUi.TopBar.GetIndex() + 1);
		globalUi.MoveChild(globalUi.CardPreviewContainer, globalUi.TopBar.GetIndex() + 1);
		globalUi.MoveChild(globalUi.MessyCardPreviewContainer, globalUi.TopBar.GetIndex() + 1);
		base.Visible = false;
	}
}
