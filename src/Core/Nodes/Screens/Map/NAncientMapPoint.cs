using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Map;

public partial class NAncientMapPoint : NMapPoint
{
	private TextureRect _icon;

	private TextureRect _outline;

	private Tween? _tween;

	private const float _pulseSpeed = 4f;

	private const float _scaleAmount = 0.05f;

	private const float _scaleBase = 1f;

	private float _elapsedTime = Rng.Chaotic.NextFloat(3140f);

	protected override Color TraveledColor => StsColors.pathDotTraveled;

	protected override Color UntravelableColor => StsColors.bossNodeUntraveled;

	protected override Color HoveredColor => StsColors.pathDotTraveled;

	protected override Vector2 HoverScale => Vector2.One * 1.1f;

	protected override Vector2 DownScale => Vector2.One * 0.9f;

	private static string UntravelableMaterialPath => "res://materials/boss_map_point_unavailable.tres";

	private static string AncientMapPointPath => SceneHelper.GetScenePath("ui/ancient_map_point");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlyArray<string>(new string[2] { UntravelableMaterialPath, AncientMapPointPath });

	private Material? TargetMaterial
	{
		get
		{
			if (base.IsTravelable || base.State == MapPointState.Traveled)
			{
				return null;
			}
			return PreloadManager.Cache.GetMaterial(UntravelableMaterialPath);
		}
	}

	public static NAncientMapPoint Create(MapPoint point, NMapScreen screen, IRunState runState)
	{
		NAncientMapPoint nAncientMapPoint = PreloadManager.Cache.GetScene(AncientMapPointPath).Instantiate<NAncientMapPoint>(PackedScene.GenEditState.Disabled);
		nAncientMapPoint.Point = point;
		nAncientMapPoint._screen = screen;
		nAncientMapPoint._runState = runState;
		return nAncientMapPoint;
	}

	public override void _Ready()
	{
		ConnectSignals();
		_icon = GetNode<TextureRect>("Icon");
		_icon.Texture = _runState.Act.Ancient.MapIcon;
		_outline = GetNode<TextureRect>("Icon/Outline");
		_outline.Texture = _runState.Act.Ancient.MapIconOutline;
		_outline.Modulate = _runState.Act.MapBgColor;
		RefreshColorInstantly();
	}

	public override void _Process(double delta)
	{
		if (_isEnabled)
		{
			if (!base.IsFocused && IsInputAllowed())
			{
				_elapsedTime += (float)delta * 4f;
				base.Scale = Vector2.One * (Mathf.Sin(_elapsedTime) * 0.05f + 1f);
			}
			else
			{
				base.Scale = base.Scale.Lerp(Vector2.One, 0.5f);
			}
		}
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		if (IsInputAllowed())
		{
			AnimHover();
			if (NControllerManager.Instance.IsUsingController)
			{
				_controllerSelectionReticle.OnSelect();
			}
		}
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		if (IsInputAllowed())
		{
			AnimUnhover();
			_controllerSelectionReticle.OnDeselect();
		}
	}

	protected override void OnPress()
	{
		if (base.IsTravelable)
		{
			AnimPressDown();
			_controllerSelectionReticle.OnDeselect();
		}
	}

	public override void OnSelected()
	{
		base.State = MapPointState.Traveled;
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_icon, "scale", Vector2.One, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back);
		_tween.TweenProperty(_icon, "self_modulate", base.TargetColor, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(_outline, "modulate", _runState.Act.MapBgColor, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	protected override void RefreshColorInstantly()
	{
		_icon.SelfModulate = base.TargetColor;
	}

	private void AnimHover()
	{
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_icon, "scale", HoverScale, 0.05);
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
		_tween.TweenProperty(_icon, "self_modulate", base.TargetColor, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(_outline, "modulate", _runState.Act.MapBgColor, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	private void AnimPressDown()
	{
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_icon, "scale", DownScale, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		_tween.TweenProperty(_outline, "modulate", _runState.Act.MapBgColor, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}
}
