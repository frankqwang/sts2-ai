using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Debug;

public partial class NSceneBootstrapper : Node
{
	private bool _openConsole;

	private NGame _game;

	public override void _Ready()
	{
		if (GetParent() is NGame game)
		{
			_game = game;
		}
		else
		{
			_game = SceneHelper.Instantiate<NGame>("game");
			_game.StartOnMainMenu = false;
			this.AddChildSafely(_game);
		}
		TaskHelper.RunSafely(StartNewRun());
	}

	private async Task StartNewRun()
	{
		Type type = BootstrapSettingsUtil.Get();
		if (type == null)
		{
			Log.Error("No type implementing IBootstrapSettings found in the project! To use the bootstrap scene, copy src/Core/Nodes/Debug/BootstrapSettings.cs.template and rename it to BootstrapSettings.cs");
			return;
		}
		IBootstrapSettings settings = (IBootstrapSettings)Activator.CreateInstance(type);
		if (settings.Language != null)
		{
			LocManager.Instance.SetLanguage(settings.Language);
		}
		PreloadManager.Enabled = settings.DoPreloading;
		string seed = settings.Seed ?? SeedHelper.GetRandomSeed();
		List<ActModel> list = ActModel.GetDefaultList().ToList();
		list[0] = settings.Act;
		RunState runState = RunState.CreateForNewRun(new global::_003C_003Ez__ReadOnlySingleElementList<Player>(Player.CreateForNewRun(settings.Character, SaveManager.Instance.GenerateUnlockStateFromProgress(), 1uL)), list.Select((ActModel a) => a.ToMutable()).ToList(), settings.Modifiers, settings.Ascension, seed);
		RunManager.Instance.SetUpNewSinglePlayer(runState, settings.SaveRunHistory);
		await PreloadManager.LoadRunAssets(new global::_003C_003Ez__ReadOnlySingleElementList<CharacterModel>(settings.Character));
		RunManager.Instance.Launch();
		_game.RootSceneContainer.SetCurrentScene(NRun.Create(runState));
		await RunManager.Instance.SetActInternal(0);
		RunManager.Instance.RunLocationTargetedBuffer.OnRunLocationChanged(runState.CurrentLocation);
		RunManager.Instance.MapSelectionSynchronizer.OnRunLocationChanged(runState.CurrentLocation);
		await settings.Setup(runState.Players[0]);
		switch (settings.RoomType)
		{
		case RoomType.Unassigned:
			await RunManager.Instance.EnterAct(0);
			break;
		case RoomType.Treasure:
		case RoomType.Shop:
		case RoomType.RestSite:
			await RunManager.Instance.EnterRoomDebug(settings.RoomType);
			RunManager.Instance.ActionExecutor.Unpause();
			break;
		case RoomType.Event:
		{
			AbstractRoom abstractRoom = await RunManager.Instance.EnterRoomDebug(settings.RoomType, MapPointType.Unassigned, settings.Event);
			if (abstractRoom != null && abstractRoom.IsVictoryRoom)
			{
				runState.CurrentActIndex = runState.Acts.Count - 1;
			}
			break;
		}
		default:
			await RunManager.Instance.EnterRoomDebug(settings.RoomType, MapPointType.Unassigned, settings.RoomType.IsCombatRoom() ? settings.Encounter.ToMutable() : null);
			break;
		}
		if (_openConsole)
		{
			NDevConsole node = GetNode<NDevConsole>("/root/DevConsole/ConsoleScreen");
			node.ShowConsole();
			node.MakeFullScreen();
			node.SetBackgroundColor(Colors.White);
		}
	}
}
