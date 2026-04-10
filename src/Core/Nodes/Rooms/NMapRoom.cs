using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Screens.Capstones;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Rooms;

public partial class NMapRoom : Control, IScreenContext
{
	private const string _scenePath = "res://scenes/rooms/map_room.tscn";

	private ActModel _act;

	private int _actIndex;

	public static IEnumerable<string> AssetPaths
	{
		get
		{
			List<string> list = new List<string>();
			list.Add("res://scenes/rooms/map_room.tscn");
			list.AddRange(NMapScreen.AssetPaths);
			return new _003C_003Ez__ReadOnlyList<string>(list);
		}
	}

	public Control? DefaultFocusedControl => null;

	public static NMapRoom? Create(ActModel act, int actIndex)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NMapRoom nMapRoom = PreloadManager.Cache.GetScene("res://scenes/rooms/map_room.tscn").Instantiate<NMapRoom>(PackedScene.GenEditState.Disabled);
		nMapRoom._act = act;
		nMapRoom._actIndex = actIndex;
		return nMapRoom;
	}

	public override void _Ready()
	{
		NMapScreen nMapScreen = NMapScreen.Instance.Open();
		NRun.Instance.GlobalUi.TopBar.Map.Disable();
		nMapScreen.SetTravelEnabled(enabled: true);
		NCapstoneContainer.Instance.Connect(NCapstoneContainer.SignalName.CapstoneClosed, Callable.From(ReopenMap));
		NActBanner child = NActBanner.Create(_act, _actIndex);
		NRun.Instance.GlobalUi.MapScreen.AddChildSafely(child);
	}

	private void ReopenMap()
	{
		NMapScreen.Instance.Open();
	}

	public override void _ExitTree()
	{
		NCapstoneContainer.Instance.Disconnect(NCapstoneContainer.SignalName.CapstoneClosed, Callable.From(ReopenMap));
		NCapstoneContainer.Instance.Close();
		NRun.Instance?.GlobalUi.TopBar.Map.Enable();
	}
}
