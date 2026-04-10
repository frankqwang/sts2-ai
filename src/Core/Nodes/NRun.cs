using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Multiplayer.Game.PeerInput;
using MegaCrit.Sts2.Core.Nodes.Audio;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.Capstones;
using MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes;

public partial class NRun : Control
{
	private const string _scenePath = "res://scenes/run.tscn";

	[Export(PropertyHint.None, "")]
	private PackedScene _cardScene;

	private RunState _state;

	private NSceneContainer _roomContainer;

	private Button _testButton;

	private MegaLabel _seedLabel;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>("res://scenes/run.tscn");

	public static NRun? Instance => NGame.Instance?.CurrentRunNode;

	public NCombatRoom? CombatRoom
	{
		get
		{
			Control currentScene = _roomContainer.CurrentScene;
			if (!(currentScene is NCombatRoom result))
			{
				if (currentScene is NEventRoom nEventRoom)
				{
					return nEventRoom.EmbeddedCombatRoom;
				}
				return null;
			}
			return result;
		}
	}

	public NTreasureRoom? TreasureRoom => _roomContainer.CurrentScene as NTreasureRoom;

	public NEventRoom? EventRoom => _roomContainer.CurrentScene as NEventRoom;

	public NRestSiteRoom? RestSiteRoom => _roomContainer.CurrentScene as NRestSiteRoom;

	public NMapRoom? MapRoom => _roomContainer.CurrentScene as NMapRoom;

	public NMerchantRoom? MerchantRoom => _roomContainer.CurrentScene as NMerchantRoom;

	public NGlobalUi GlobalUi { get; private set; }

	public NRunMusicController RunMusicController { get; private set; }

	public ScreenStateTracker ScreenStateTracker { get; private set; }

	public static NRun Create(RunState state)
	{
		NRun nRun = PreloadManager.Cache.GetScene("res://scenes/run.tscn").Instantiate<NRun>(PackedScene.GenEditState.Disabled);
		nRun._state = state;
		return nRun;
	}

	public override void _Ready()
	{
		_roomContainer = GetNode<NSceneContainer>("%RoomContainer");
		GlobalUi = GetNode<NGlobalUi>("%GlobalUi");
		GlobalUi.Initialize(_state);
		ScreenStateTracker = new ScreenStateTracker(GlobalUi.MapScreen, GlobalUi.CapstoneContainer, GlobalUi.Overlays);
		RunMusicController = GetNode<NRunMusicController>("%RunMusicController");
		RunMusicController.SetRunState(_state);
		RunMusicController.UpdateMusic();
		_seedLabel = GetNode<MegaLabel>("%DebugSeed");
		_seedLabel.SetTextAutoSize(_state.Rng.StringSeed);
	}

	public override void _Process(double delta)
	{
		RunManager.Instance.NetService.Update();
	}

	public override void _Notification(int what)
	{
		if ((long)what == 1006)
		{
			RunManager.Instance.CleanUp(graceful: false);
		}
	}

	public void SetCurrentRoom(Control? node)
	{
		if (node != null)
		{
			_roomContainer.SetCurrentScene(node);
			ActiveScreenContext.Instance.Update();
		}
	}

	public void ShowGameOverScreen(SerializableRun serializableRun)
	{
		NCapstoneContainer.Instance.Close();
		NMapScreen.Instance.Close();
		NOverlayStack.Instance.Push(NGameOverScreen.Create(_state, serializableRun));
	}
}
