using System;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.sts2.Core.Nodes.TopBar;

public partial class NTopBarRoomIcon : NClickableControl
{
	private IRunState _runState;

	private TextureRect _roomIcon;

	private TextureRect _roomIconOutline;

	private MapPointType _debugMapPointTypeOverride;

	public override void _Ready()
	{
		_roomIcon = GetNode<TextureRect>("Icon");
		_roomIconOutline = GetNode<TextureRect>("Icon/Outline");
		ConnectSignals();
	}

	public override void _EnterTree()
	{
		RunManager.Instance.RoomEntered += UpdateIcon;
	}

	public override void _ExitTree()
	{
		RunManager.Instance.RoomEntered -= UpdateIcon;
	}

	public void Initialize(IRunState runState)
	{
		_runState = runState;
		UpdateIcon();
	}

	public void DebugSetMapPointTypeOverride(MapPointType mapPointType)
	{
		if (mapPointType != MapPointType.Unassigned)
		{
			_debugMapPointTypeOverride = mapPointType;
		}
	}

	public void DebugClearMapPointTypeOverride()
	{
		_debugMapPointTypeOverride = MapPointType.Unassigned;
	}

	protected override void OnFocus()
	{
		string hoverTipPrefixForRoomType = GetHoverTipPrefixForRoomType();
		NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(hoverTip: new HoverTip(new LocString("static_hover_tips", hoverTipPrefixForRoomType + ".title"), new LocString("static_hover_tips", hoverTipPrefixForRoomType + ".description")), owner: _roomIcon);
		nHoverTipSet.GlobalPosition = _roomIcon.GlobalPosition + new Vector2(0f, base.Size.Y + 20f);
	}

	private string GetHoverTipPrefixForRoomType()
	{
		return GetCurrentMapPointType() switch
		{
			MapPointType.Unassigned => "ROOM_MAP", 
			MapPointType.Unknown => GetHoverTipPrefixForUnknownRoomType(), 
			MapPointType.Shop => "ROOM_MERCHANT", 
			MapPointType.Treasure => "ROOM_TREASURE", 
			MapPointType.RestSite => "ROOM_REST", 
			MapPointType.Monster => "ROOM_ENEMY", 
			MapPointType.Elite => "ROOM_ELITE", 
			MapPointType.Boss => "ROOM_BOSS", 
			MapPointType.Ancient => "ROOM_ANCIENT", 
			_ => throw new ArgumentOutOfRangeException(), 
		};
	}

	private string GetHoverTipPrefixForUnknownRoomType()
	{
		AbstractRoom baseRoom = _runState.BaseRoom;
		return baseRoom.RoomType switch
		{
			RoomType.Monster => "ROOM_UNKNOWN_ENEMY", 
			RoomType.Treasure => "ROOM_UNKNOWN_TREASURE", 
			RoomType.Shop => "ROOM_UNKNOWN_MERCHANT", 
			RoomType.Event => "ROOM_UNKNOWN_EVENT", 
			_ => throw new ArgumentOutOfRangeException(), 
		};
	}

	protected override void OnUnfocus()
	{
		NHoverTipSet.Remove(_roomIcon);
	}

	private void UpdateIcon()
	{
		if (_runState.CurrentRoom != null)
		{
			AbstractRoom baseRoom = _runState.BaseRoom;
			ActModel act = _runState.Act;
			MapPointType currentMapPointType = GetCurrentMapPointType();
			ModelId modelId = null;
			switch (currentMapPointType)
			{
			case MapPointType.Boss:
				modelId = ((_runState.CurrentMapPoint == _runState.Map.SecondBossMapPoint) ? act.SecondBossEncounter.Id : act.BossEncounter.Id);
				break;
			case MapPointType.Ancient:
				modelId = act.Ancient.Id;
				break;
			}
			string roomIconPath = ImageHelper.GetRoomIconPath(currentMapPointType, baseRoom.RoomType, modelId);
			if (roomIconPath != null)
			{
				_roomIcon.Visible = true;
				_roomIcon.Texture = PreloadManager.Cache.GetCompressedTexture2D(roomIconPath);
			}
			else
			{
				_roomIcon.Visible = false;
			}
			string roomIconOutlinePath = ImageHelper.GetRoomIconOutlinePath(currentMapPointType, baseRoom.RoomType, modelId);
			if (roomIconOutlinePath != null)
			{
				_roomIconOutline.Visible = true;
				_roomIconOutline.Texture = PreloadManager.Cache.GetCompressedTexture2D(roomIconOutlinePath);
			}
			else
			{
				_roomIconOutline.Visible = false;
			}
			if (baseRoom.IsVictoryRoom)
			{
				_roomIcon.Visible = false;
				_roomIconOutline.Visible = false;
				base.FocusMode = FocusModeEnum.None;
				base.MouseFilter = MouseFilterEnum.Ignore;
			}
		}
	}

	private MapPointType GetCurrentMapPointType()
	{
		if (_debugMapPointTypeOverride == MapPointType.Unassigned)
		{
			return _runState.CurrentMapPoint?.PointType ?? MapPointType.Unassigned;
		}
		return _debugMapPointTypeOverride;
	}
}
