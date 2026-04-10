using System;
using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Screens.Bestiary;
using MegaCrit.Sts2.Core.Nodes.Screens.CardLibrary;
using MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;
using MegaCrit.Sts2.Core.Nodes.Screens.CustomRun;
using MegaCrit.Sts2.Core.Nodes.Screens.DailyRun;
using MegaCrit.Sts2.Core.Nodes.Screens.ModdingScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.PotionLab;
using MegaCrit.Sts2.Core.Nodes.Screens.ProfileScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.RelicCollection;
using MegaCrit.Sts2.Core.Nodes.Screens.RunHistoryScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.Settings;
using MegaCrit.Sts2.Core.Nodes.Screens.StatsScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.Timeline;

namespace MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;

public partial class NMainMenuSubmenuStack : NSubmenuStack
{
	[Export(PropertyHint.None, "")]
	private PackedScene _settingsScreenScene;

	[Export(PropertyHint.None, "")]
	private PackedScene _characterSelectScreenScene;

	private NSingleplayerSubmenu? _singleplayerSubmenu;

	private NMultiplayerSubmenu? _multiplayerSubmenu;

	private NMultiplayerHostSubmenu? _multiplayerHostSubmenu;

	private NJoinFriendScreen? _joinFriendSubmenu;

	private NCharacterSelectScreen? _characterSelectSubmenu;

	private NMultiplayerLoadGameScreen? _loadMultiplayerSubmenu;

	private NCompendiumSubmenu? _compendiumSubmenu;

	private NBestiary? _bestiarySubmenu;

	private NRelicCollection? _relicCollectionSubmenu;

	private NPotionLab? _potionLabSubmenu;

	private NCardLibrary? _cardLibrarySubmenu;

	private NRunHistory? _runHistorySubmenu;

	private NStatsScreen? _statsScreen;

	private NTimelineScreen? _timelineScreen;

	private NSettingsScreen? _settingsScreen;

	private NDailyRunScreen? _dailyScreen;

	private NDailyRunLoadScreen? _dailyLoadScreen;

	private NCustomRunScreen? _customRunScreen;

	private NCustomRunLoadScreen? _customRunLoadScreen;

	private NModdingScreen? _moddingScreen;

	private NProfileScreen? _profileScreen;

	public override void _Ready()
	{
		GetSubmenuType<NSettingsScreen>();
		GetSubmenuType<NCharacterSelectScreen>();
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
		if (type == typeof(NSingleplayerSubmenu))
		{
			if (_singleplayerSubmenu == null)
			{
				_singleplayerSubmenu = NSingleplayerSubmenu.Create();
				_singleplayerSubmenu.Visible = false;
				this.AddChildSafely(_singleplayerSubmenu);
			}
			return _singleplayerSubmenu;
		}
		if (type == typeof(NMultiplayerSubmenu))
		{
			if (_multiplayerSubmenu == null)
			{
				_multiplayerSubmenu = NMultiplayerSubmenu.Create();
				_multiplayerSubmenu.Visible = false;
				this.AddChildSafely(_multiplayerSubmenu);
			}
			return _multiplayerSubmenu;
		}
		if (type == typeof(NMultiplayerHostSubmenu))
		{
			if (_multiplayerHostSubmenu == null)
			{
				_multiplayerHostSubmenu = NMultiplayerHostSubmenu.Create();
				_multiplayerHostSubmenu.Visible = false;
				this.AddChildSafely(_multiplayerHostSubmenu);
			}
			return _multiplayerHostSubmenu;
		}
		if (type == typeof(NJoinFriendScreen))
		{
			if (_joinFriendSubmenu == null)
			{
				_joinFriendSubmenu = NJoinFriendScreen.Create();
				_joinFriendSubmenu.Visible = false;
				this.AddChildSafely(_joinFriendSubmenu);
			}
			return _joinFriendSubmenu;
		}
		if (type == typeof(NCharacterSelectScreen))
		{
			if (_characterSelectSubmenu == null)
			{
				_characterSelectSubmenu = _characterSelectScreenScene.Instantiate<NCharacterSelectScreen>(PackedScene.GenEditState.Disabled);
				_characterSelectSubmenu.Visible = false;
				this.AddChildSafely(_characterSelectSubmenu);
			}
			return _characterSelectSubmenu;
		}
		if (type == typeof(NMultiplayerLoadGameScreen))
		{
			if (_loadMultiplayerSubmenu == null)
			{
				_loadMultiplayerSubmenu = NMultiplayerLoadGameScreen.Create();
				_loadMultiplayerSubmenu.Visible = false;
				this.AddChildSafely(_loadMultiplayerSubmenu);
			}
			return _loadMultiplayerSubmenu;
		}
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
		if (type == typeof(NMultiplayerHostSubmenu))
		{
			if (_multiplayerHostSubmenu == null)
			{
				_multiplayerHostSubmenu = NMultiplayerHostSubmenu.Create();
				_multiplayerHostSubmenu.Visible = false;
				this.AddChildSafely(_multiplayerHostSubmenu);
			}
			return _multiplayerHostSubmenu;
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
			if (_runHistorySubmenu == null)
			{
				_runHistorySubmenu = NRunHistory.Create();
				_runHistorySubmenu.Visible = false;
				this.AddChildSafely(_runHistorySubmenu);
			}
			return _runHistorySubmenu;
		}
		if (type == typeof(NStatsScreen))
		{
			if (_statsScreen == null)
			{
				_statsScreen = NStatsScreen.Create();
				_statsScreen.Visible = false;
				this.AddChildSafely(_statsScreen);
			}
			return _statsScreen;
		}
		if (type == typeof(NTimelineScreen))
		{
			if (_timelineScreen == null)
			{
				_timelineScreen = NTimelineScreen.Create();
				_timelineScreen.Visible = false;
				this.AddChildSafely(_timelineScreen);
			}
			return _timelineScreen;
		}
		if (type == typeof(NSettingsScreen))
		{
			if (_settingsScreen == null)
			{
				_settingsScreen = _settingsScreenScene.Instantiate<NSettingsScreen>(PackedScene.GenEditState.Disabled);
				_settingsScreen.SetIsInRun(isInRun: false);
				_settingsScreen.Visible = false;
				this.AddChildSafely(_settingsScreen);
			}
			return _settingsScreen;
		}
		if (type == typeof(NDailyRunScreen))
		{
			if (_dailyScreen == null)
			{
				_dailyScreen = NDailyRunScreen.Create();
				_dailyScreen.Visible = false;
				this.AddChildSafely(_dailyScreen);
			}
			return _dailyScreen;
		}
		if (type == typeof(NDailyRunLoadScreen))
		{
			if (_dailyLoadScreen == null)
			{
				_dailyLoadScreen = NDailyRunLoadScreen.Create();
				_dailyLoadScreen.Visible = false;
				this.AddChildSafely(_dailyLoadScreen);
			}
			return _dailyLoadScreen;
		}
		if (type == typeof(NCustomRunScreen))
		{
			if (_customRunScreen == null)
			{
				_customRunScreen = NCustomRunScreen.Create();
				_customRunScreen.Visible = false;
				this.AddChildSafely(_customRunScreen);
			}
			return _customRunScreen;
		}
		if (type == typeof(NCustomRunLoadScreen))
		{
			if (_customRunLoadScreen == null)
			{
				_customRunLoadScreen = NCustomRunLoadScreen.Create();
				_customRunLoadScreen.Visible = false;
				this.AddChildSafely(_customRunLoadScreen);
			}
			return _customRunLoadScreen;
		}
		if (type == typeof(NModdingScreen))
		{
			if (_moddingScreen == null)
			{
				_moddingScreen = NModdingScreen.Create();
				_moddingScreen.Visible = false;
				this.AddChildSafely(_moddingScreen);
			}
			return _moddingScreen;
		}
		if (type == typeof(NProfileScreen))
		{
			if (_profileScreen == null)
			{
				_profileScreen = NProfileScreen.Create();
				_profileScreen.Visible = false;
				this.AddChildSafely(_profileScreen);
			}
			return _profileScreen;
		}
		throw new ArgumentException($"No such submenu {type} in main menu");
	}
}
