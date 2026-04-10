using System;
using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Screens.Bestiary;
using MegaCrit.Sts2.Core.Nodes.Screens.CardLibrary;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Nodes.Screens.PauseMenu;
using MegaCrit.Sts2.Core.Nodes.Screens.PotionLab;
using MegaCrit.Sts2.Core.Nodes.Screens.RelicCollection;
using MegaCrit.Sts2.Core.Nodes.Screens.RunHistoryScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.Settings;
using MegaCrit.Sts2.Core.Nodes.Screens.StatsScreen;

namespace MegaCrit.Sts2.Core.Nodes.Screens;

public partial class NRunSubmenuStack : NSubmenuStack
{
	[Export(PropertyHint.None, "")]
	private PackedScene _settingsScreenScene;

	[Export(PropertyHint.None, "")]
	private PackedScene _pauseMenuScene;

	[Export(PropertyHint.None, "")]
	private PackedScene _statsScreenScene;

	[Export(PropertyHint.None, "")]
	private PackedScene _runHistoryScreenScene;

	private NCompendiumSubmenu? _compendiumSubmenu;

	private NBestiary? _bestiarySubmenu;

	private NRelicCollection? _relicCollectionSubmenu;

	private NPotionLab? _potionLabSubmenu;

	private NCardLibrary? _cardLibrarySubmenu;

	private NRunHistory? _runHistoryScreen;

	private NSettingsScreen? _settingsScreen;

	private NStatsScreen? _statsScreen;

	private NPauseMenu? _pauseMenu;

	public override void _Ready()
	{
		GetSubmenuType<NSettingsScreen>();
	}

	public override T PushSubmenuType<T>()
	{
		return (T)PushSubmenuType(typeof(T));
	}

	public override T GetSubmenuType<T>()
	{
		return (T)GetSubmenuType(typeof(T));
	}

	public override NSubmenu PushSubmenuType(Type type)
	{
		NSubmenu submenuType = GetSubmenuType(type);
		Push(submenuType);
		return submenuType;
	}

	public override NSubmenu GetSubmenuType(Type type)
	{
		if (type == typeof(NCompendiumSubmenu))
		{
			if (_compendiumSubmenu == null)
			{
				_compendiumSubmenu = NCompendiumSubmenu.Create();
				_compendiumSubmenu.Visible = false;
				this.AddChildSafely(_compendiumSubmenu);
			}
			return _compendiumSubmenu;
		}
		if (type == typeof(NBestiary))
		{
			if (_bestiarySubmenu == null)
			{
				_bestiarySubmenu = NBestiary.Create();
				_bestiarySubmenu.Visible = false;
				this.AddChildSafely(_bestiarySubmenu);
			}
			return _bestiarySubmenu;
		}
		if (type == typeof(NRelicCollection))
		{
			if (_relicCollectionSubmenu == null)
			{
				_relicCollectionSubmenu = NRelicCollection.Create();
				_relicCollectionSubmenu.Visible = false;
				this.AddChildSafely(_relicCollectionSubmenu);
			}
			return _relicCollectionSubmenu;
		}
		if (type == typeof(NPotionLab))
		{
			if (_potionLabSubmenu == null)
			{
				_potionLabSubmenu = NPotionLab.Create();
				_potionLabSubmenu.Visible = false;
				this.AddChildSafely(_potionLabSubmenu);
			}
			return _potionLabSubmenu;
		}
		if (type == typeof(NCardLibrary))
		{
			if (_cardLibrarySubmenu == null)
			{
				_cardLibrarySubmenu = NCardLibrary.Create();
				_cardLibrarySubmenu.Visible = false;
				this.AddChildSafely(_cardLibrarySubmenu);
			}
			return _cardLibrarySubmenu;
		}
		if (type == typeof(NRunHistory))
		{
			if (_runHistoryScreen == null)
			{
				_runHistoryScreen = _runHistoryScreenScene.Instantiate<NRunHistory>(PackedScene.GenEditState.Disabled);
				_runHistoryScreen.Visible = false;
				this.AddChildSafely(_runHistoryScreen);
			}
			return _runHistoryScreen;
		}
		if (type == typeof(NSettingsScreen))
		{
			if (_settingsScreen == null)
			{
				_settingsScreen = _settingsScreenScene.Instantiate<NSettingsScreen>(PackedScene.GenEditState.Disabled);
				_settingsScreen.SetIsInRun(isInRun: true);
				_settingsScreen.Visible = false;
				this.AddChildSafely(_settingsScreen);
			}
			return _settingsScreen;
		}
		if (type == typeof(NStatsScreen))
		{
			if (_statsScreen == null)
			{
				_statsScreen = _statsScreenScene.Instantiate<NStatsScreen>(PackedScene.GenEditState.Disabled);
				_statsScreen.Visible = false;
				this.AddChildSafely(_statsScreen);
			}
			return _statsScreen;
		}
		if (type == typeof(NPauseMenu))
		{
			if (_pauseMenu == null)
			{
				_pauseMenu = _pauseMenuScene.Instantiate<NPauseMenu>(PackedScene.GenEditState.Disabled);
				_pauseMenu.Visible = false;
				this.AddChildSafely(_pauseMenu);
			}
			return _pauseMenu;
		}
		throw new ArgumentException($"No such submenu of type {type} in run");
	}
}
