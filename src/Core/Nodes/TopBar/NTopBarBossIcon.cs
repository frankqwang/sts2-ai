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

public partial class NTopBarBossIcon : NClickableControl
{
	private static readonly LocString _bossHoverTipTitle = new LocString("static_hover_tips", "BOSS.title");

	private static readonly LocString _bossHoverTipDescription = new LocString("static_hover_tips", "BOSS.description");

	private static readonly LocString _doubleBossHoverTipTitle = new LocString("static_hover_tips", "DOUBLE_BOSS.title");

	private static readonly LocString _doubleBossHoverTipDescription = new LocString("static_hover_tips", "DOUBLE_BOSS.description");

	private TextureRect _bossIcon;

	private TextureRect _bossIconOutline;

	private TextureRect? _secondBossIcon;

	private TextureRect? _secondBossIconOutline;

	private static readonly StringName _tintColor = new StringName("tint_color");

	private const string _secondBossIconScenePath = "res://scenes/ui/top_bar/second_boss_icon.tscn";

	private IRunState _runState;

	private bool ShouldOnlyShowSecondBossIcon
	{
		get
		{
			if (_runState.Map.SecondBossMapPoint != null)
			{
				return _runState.CurrentMapPoint == _runState.Map.BossMapPoint;
			}
			return false;
		}
	}

	public override void _Ready()
	{
		_bossIcon = GetNode<TextureRect>("Icon");
		_bossIconOutline = GetNode<TextureRect>("Icon/Outline");
		ConnectSignals();
	}

	public void Initialize(IRunState runState)
	{
		_runState = runState;
		OnActEntered();
	}

	public override void _EnterTree()
	{
		RunManager.Instance.ActEntered += OnActEntered;
		RunManager.Instance.RoomEntered += OnRoomEntered;
	}

	public override void _ExitTree()
	{
		RunManager.Instance.ActEntered -= OnActEntered;
		RunManager.Instance.RoomEntered -= OnRoomEntered;
	}

	private void OnRoomEntered()
	{
		if (_runState.CurrentRoom != null)
		{
			AbstractRoom baseRoom = _runState.BaseRoom;
			base.Visible = baseRoom.RoomType != RoomType.Boss || ShouldOnlyShowSecondBossIcon;
			base.FocusMode = (FocusModeEnum)((baseRoom.RoomType != RoomType.Boss || ShouldOnlyShowSecondBossIcon) ? 2 : 0);
			if (ShouldOnlyShowSecondBossIcon)
			{
				RefreshBossIcon();
			}
			RefreshSecondBossIconColor();
			if (_runState.CurrentRoom.IsVictoryRoom)
			{
				_bossIcon.SetVisible(visible: false);
				_bossIconOutline.SetVisible(visible: false);
				_secondBossIcon?.SetVisible(visible: false);
				_secondBossIconOutline?.SetVisible(visible: false);
				base.FocusMode = FocusModeEnum.None;
				base.MouseFilter = MouseFilterEnum.Ignore;
			}
		}
	}

	private void OnActEntered()
	{
		RefreshBossIcon();
	}

	public void RefreshBossIcon()
	{
		EncounterModel encounterModel = (ShouldOnlyShowSecondBossIcon ? _runState.Act.SecondBossEncounter : _runState.Act.BossEncounter);
		string roomIconPath = ImageHelper.GetRoomIconPath(MapPointType.Boss, RoomType.Boss, encounterModel.Id);
		string roomIconOutlinePath = ImageHelper.GetRoomIconOutlinePath(MapPointType.Boss, RoomType.Boss, encounterModel.Id);
		_bossIcon.Texture = PreloadManager.Cache.GetTexture2D(roomIconPath);
		_bossIconOutline.Texture = PreloadManager.Cache.GetTexture2D(roomIconOutlinePath);
		EncounterModel secondBossEncounter = _runState.Act.SecondBossEncounter;
		if (secondBossEncounter != null && !ShouldOnlyShowSecondBossIcon)
		{
			if (_secondBossIcon == null)
			{
				PackedScene packedScene = GD.Load<PackedScene>("res://scenes/ui/top_bar/second_boss_icon.tscn");
				_secondBossIcon = packedScene.Instantiate<TextureRect>(PackedScene.GenEditState.Disabled);
				_secondBossIconOutline = _secondBossIcon.GetNode<TextureRect>("%Outline");
				_secondBossIcon.MouseFilter = MouseFilterEnum.Pass;
				_secondBossIconOutline.MouseFilter = MouseFilterEnum.Pass;
				_bossIcon.AddChildSafely(_secondBossIcon);
				_secondBossIcon.Position = new Vector2(30f, 22f);
			}
			string roomIconPath2 = ImageHelper.GetRoomIconPath(MapPointType.Boss, RoomType.Boss, secondBossEncounter.Id);
			string roomIconOutlinePath2 = ImageHelper.GetRoomIconOutlinePath(MapPointType.Boss, RoomType.Boss, secondBossEncounter.Id);
			_secondBossIcon.Texture = PreloadManager.Cache.GetTexture2D(roomIconPath2);
			_secondBossIconOutline.Texture = PreloadManager.Cache.GetTexture2D(roomIconOutlinePath2);
			_secondBossIcon.Visible = true;
			RefreshSecondBossIconColor();
		}
		else
		{
			_secondBossIcon?.SetVisible(visible: false);
			_secondBossIconOutline?.SetVisible(visible: false);
		}
	}

	private void RefreshSecondBossIconColor()
	{
		if (_secondBossIcon?.Material is ShaderMaterial shaderMaterial && _secondBossIconOutline?.Material is ShaderMaterial shaderMaterial2)
		{
			ActModel act = _runState.Act;
			MapPoint currentMapPoint = _runState.CurrentMapPoint;
			MapPoint bossMapPoint = _runState.Map.BossMapPoint;
			MapPoint secondBossMapPoint = _runState.Map.SecondBossMapPoint;
			Color color = ((currentMapPoint == bossMapPoint || currentMapPoint == secondBossMapPoint) ? act.MapTraveledColor : act.MapUntraveledColor);
			shaderMaterial.SetShaderParameter(_tintColor, new Vector3(color.R, color.G, color.B));
			shaderMaterial2.SetShaderParameter(_tintColor, new Vector3(color.R, color.G, color.B));
		}
	}

	protected override void OnFocus()
	{
		EncounterModel bossEncounter = _runState.Act.BossEncounter;
		EncounterModel secondBossEncounter = _runState.Act.SecondBossEncounter;
		HoverTip hoverTip;
		if (secondBossEncounter != null && !ShouldOnlyShowSecondBossIcon)
		{
			_doubleBossHoverTipTitle.Add("BossName1", bossEncounter.Title);
			_doubleBossHoverTipTitle.Add("BossName2", secondBossEncounter.Title);
			_doubleBossHoverTipDescription.Add("BossName1", bossEncounter.Title);
			_doubleBossHoverTipDescription.Add("BossName2", secondBossEncounter.Title);
			hoverTip = new HoverTip(_doubleBossHoverTipTitle, _doubleBossHoverTipDescription);
		}
		else
		{
			_bossHoverTipTitle.Add("BossName", ShouldOnlyShowSecondBossIcon ? secondBossEncounter.Title : bossEncounter.Title);
			_bossHoverTipDescription.Add("BossName", ShouldOnlyShowSecondBossIcon ? secondBossEncounter.Title : bossEncounter.Title);
			hoverTip = new HoverTip(_bossHoverTipTitle, _bossHoverTipDescription);
		}
		NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, hoverTip);
		nHoverTipSet.GlobalPosition = _bossIcon.GlobalPosition + new Vector2(0f, base.Size.Y + 20f);
	}

	protected override void OnUnfocus()
	{
		NHoverTipSet.Remove(this);
	}
}
