using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Vfx;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Runs.History;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Map;

public partial class NNormalMapPoint : NMapPoint
{
	private static readonly StringName _mapColor = new StringName("map_color");

	private Control _iconContainer;

	private TextureRect _icon;

	private TextureRect _questIcon;

	private TextureRect _outline;

	private NMapCircleVfx? _circleVfx;

	private Tween? _tween;

	private Tween? _pulseTween;

	private const float _pulseSpeed = 4f;

	private const float _scaleAmount = 0.25f;

	private const float _scaleBase = 1.2f;

	private float _elapsedTime = Rng.Chaotic.NextFloat(3140f);

	protected override Color TraveledColor => Colors.White;

	protected override Color UntravelableColor => StsColors.halfTransparentWhite;

	protected override Color HoveredColor => Colors.White;

	protected override Vector2 HoverScale => Vector2.One * 1.45f;

	protected override Vector2 DownScale => Vector2.One * 0.9f;

	private static string ScenePath => SceneHelper.GetScenePath("/ui/normal_map_point");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	private static string IconName(MapPointType pointType)
	{
		return pointType switch
		{
			MapPointType.Unassigned => "map_unknown", 
			MapPointType.Monster => "map_monster", 
			MapPointType.Elite => "map_elite", 
			MapPointType.Boss => string.Empty, 
			MapPointType.Ancient => "map_unknown", 
			MapPointType.Treasure => "map_chest", 
			MapPointType.Shop => "map_shop", 
			MapPointType.RestSite => "map_rest", 
			MapPointType.Unknown => "map_unknown", 
			_ => throw new ArgumentOutOfRangeException(pointType.ToString()), 
		};
	}

	private static string IconPath(string filename)
	{
		return ImageHelper.GetImagePath("atlases/ui_atlas.sprites/map/icons/" + filename + ".tres");
	}

	private static string OutlinePath(string filename)
	{
		return ImageHelper.GetImagePath("atlases/compressed.sprites/map/" + filename + "_outline.tres");
	}

	private static string UnknownIconPath(RoomType pointType)
	{
		return ImageHelper.GetImagePath("atlases/ui_atlas.sprites/map/icons/map_" + pointType switch
		{
			RoomType.Treasure => "unknown_chest", 
			RoomType.Monster => "unknown_monster", 
			RoomType.Shop => "unknown_shop", 
			RoomType.Elite => "unknown_elite", 
			_ => "unknown", 
		} + ".tres");
	}

	private static string UnknownOutlinePath(RoomType pointType)
	{
		return OutlinePath(pointType switch
		{
			RoomType.Treasure => "map_chest", 
			RoomType.Monster => "map_monster", 
			RoomType.Shop => "map_shop", 
			RoomType.Elite => "map_elite", 
			_ => "map_unknown", 
		});
	}

	public static NNormalMapPoint Create(MapPoint point, NMapScreen screen, IRunState runState)
	{
		NNormalMapPoint nNormalMapPoint = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NNormalMapPoint>(PackedScene.GenEditState.Disabled);
		nNormalMapPoint.Point = point;
		nNormalMapPoint._screen = screen;
		nNormalMapPoint._runState = runState;
		return nNormalMapPoint;
	}

	public override void _Ready()
	{
		ConnectSignals();
		_iconContainer = GetNode<Control>("%IconContainer");
		_icon = GetNode<TextureRect>("%Icon");
		_outline = GetNode<TextureRect>("%Outline");
		_questIcon = GetNode<TextureRect>("%QuestIcon");
		UpdateIcon();
		Color mapBgColor = _runState.Act.MapBgColor;
		_outline.Modulate = mapBgColor;
		ShaderMaterial shaderMaterial = (ShaderMaterial)_icon.Material;
		shaderMaterial.SetShaderParameter(_mapColor, mapBgColor.Lerp(Colors.Gray, 0.5f));
		RefreshMarkedIconVisibility();
		RefreshColorInstantly();
		Disable();
	}

	public override void _EnterTree()
	{
		base.Point.NodeMarkedChanged += RefreshMarkedIconVisibility;
		NMapScreen.Instance.PointTypeHighlighted += OnHighlightPointType;
	}

	private void RefreshMarkedIconVisibility()
	{
		_questIcon.Visible = base.Point.Quests.Count > 0;
	}

	public override void _ExitTree()
	{
		base.Point.NodeMarkedChanged -= RefreshMarkedIconVisibility;
		NMapScreen.Instance.PointTypeHighlighted -= OnHighlightPointType;
	}

	public override void _Process(double delta)
	{
		if (_isEnabled)
		{
			if (!base.IsFocused && IsInputAllowed())
			{
				_elapsedTime += (float)delta * 4f;
				_iconContainer.Scale = Vector2.One * (Mathf.Sin(_elapsedTime) * 0.25f + 1.2f);
			}
			else
			{
				_iconContainer.Scale = _iconContainer.Scale.Lerp(Vector2.One, 0.5f);
			}
		}
	}

	public override void OnSelected()
	{
		ShowCircleVfx(playAnim: true);
		base.State = MapPointState.Traveled;
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_iconContainer, "scale", Vector2.One, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		_tween.TweenProperty(_icon, "scale", Vector2.One, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back);
		_tween.TweenProperty(_questIcon, "scale", Vector2.One, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back);
		_tween.TweenProperty(_icon, "self_modulate", base.TargetColor, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(_outline, "modulate", _runState.Act.MapBgColor, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		if (IsInputAllowed())
		{
			AnimHover();
		}
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		NHoverTipSet.Remove(this);
		if (_isEnabled)
		{
			_elapsedTime = 3.926991f;
		}
		if (IsInputAllowed())
		{
			AnimUnhover();
		}
	}

	protected override void OnPress()
	{
		if (base.IsTravelable)
		{
			AnimPressDown();
			NHoverTipSet.Remove(this);
		}
	}

	public void SetAngle(float degrees)
	{
		_iconContainer.RotationDegrees = degrees;
	}

	protected override void RefreshColorInstantly()
	{
		_icon.SelfModulate = base.TargetColor;
	}

	protected override void RefreshState()
	{
		base.RefreshState();
		UpdateIcon();
		if (base.State == MapPointState.Traveled)
		{
			ShowCircleVfx(playAnim: false);
		}
		if (!base.IsFocused)
		{
			_iconContainer.Scale = Vector2.One;
		}
	}

	private void UpdateIcon()
	{
		if (base.Point.PointType != MapPointType.Unknown || base.State != MapPointState.Traveled)
		{
			string filename = IconName(base.Point.PointType);
			string path = IconPath(filename);
			_icon.Texture = ResourceLoader.Load<Texture2D>(path, null, ResourceLoader.CacheMode.Reuse);
			string path2 = OutlinePath(IconName(base.Point.PointType));
			_outline.Texture = ResourceLoader.Load<Texture2D>(path2, null, ResourceLoader.CacheMode.Reuse);
		}
		else if (_runState.MapPointHistory.Count > _runState.CurrentActIndex)
		{
			IReadOnlyList<MapPointHistoryEntry> readOnlyList = _runState.MapPointHistory[_runState.CurrentActIndex];
			if (readOnlyList.Count > base.Point.coord.row)
			{
				RoomType roomType = readOnlyList[base.Point.coord.row].Rooms.First().RoomType;
				_icon.Texture = ResourceLoader.Load<Texture2D>(UnknownIconPath(roomType), null, ResourceLoader.CacheMode.Reuse);
				_outline.Texture = ResourceLoader.Load<Texture2D>(UnknownOutlinePath(roomType), null, ResourceLoader.CacheMode.Reuse);
			}
		}
	}

	private void ShowCircleVfx(bool playAnim)
	{
		if (_circleVfx == null)
		{
			_circleVfx = NMapCircleVfx.Create(playAnim);
			this.AddChildSafely(_circleVfx);
			_circleVfx.Position += base.PivotOffset;
		}
	}

	private void AnimHover()
	{
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_icon, "scale", HoverScale, 0.05);
		_tween.TweenProperty(_questIcon, "scale", HoverScale, 0.05);
		_tween.TweenProperty(_icon, "self_modulate", HoveredColor, 0.05);
		if (base.IsTravelable)
		{
			_tween.TweenProperty(_outline, "modulate", _outlineColor, 0.05);
		}
	}

	private void AnimUnhover()
	{
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_icon, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		_tween.TweenProperty(_questIcon, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		_tween.TweenProperty(_icon, "self_modulate", base.TargetColor, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		_tween.TweenProperty(_outline, "modulate", _runState.Act.MapBgColor, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
	}

	private void AnimPressDown()
	{
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_icon, "scale", DownScale, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		_tween.TweenProperty(_questIcon, "scale", DownScale, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		_tween.TweenProperty(_outline, "modulate", _runState.Act.MapBgColor, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	private void OnHighlightPointType(MapPointType pointType)
	{
		if (pointType == base.Point.PointType)
		{
			AnimHover();
		}
		else
		{
			AnimUnhover();
		}
	}
}
