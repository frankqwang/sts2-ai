using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Map;

public partial class NBossMapPoint : NMapPoint
{
	private static readonly StringName _mapColor = new StringName("map_color");

	private static readonly StringName _blackLayerColor = new StringName("black_layer_color");

	private Tween? _hoverTween;

	private ActModel _act;

	private bool _usesSpine;

	private Node2D _spriteContainer;

	private Node2D _spineSprite;

	private MegaSprite _animController;

	private ShaderMaterial _material;

	private TextureRect _placeholderImage;

	private TextureRect _placeholderOutline;

	protected override Color TraveledColor => StsColors.pathDotTraveled;

	protected override Color UntravelableColor => StsColors.red;

	protected override Color HoveredColor => StsColors.pathDotTraveled;

	protected override Vector2 HoverScale => Vector2.One * 1.05f;

	protected override Vector2 DownScale => Vector2.One * 1.02f;

	private static string BossMapPointPath => SceneHelper.GetScenePath("ui/boss_map_point");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(BossMapPointPath);

	public static NBossMapPoint Create(MapPoint point, NMapScreen screen, IRunState runState)
	{
		NBossMapPoint nBossMapPoint = PreloadManager.Cache.GetScene(BossMapPointPath).Instantiate<NBossMapPoint>(PackedScene.GenEditState.Disabled);
		nBossMapPoint.Point = point;
		nBossMapPoint._screen = screen;
		nBossMapPoint._runState = runState;
		nBossMapPoint._act = runState.Act;
		return nBossMapPoint;
	}

	public override void _Ready()
	{
		ConnectSignals();
		Disable();
		_spriteContainer = GetNode<Node2D>("%SpriteContainer");
		_spineSprite = GetNode<Node2D>("%SpineSprite");
		_animController = new MegaSprite(_spineSprite);
		EncounterModel encounterModel = ((base.Point == _runState.Map.SecondBossMapPoint) ? _runState.Act.SecondBossEncounter : _runState.Act.BossEncounter);
		if (encounterModel.BossNodeSpineResource != null)
		{
			_usesSpine = true;
			_spineSprite.Visible = true;
			_animController.SetSkeletonDataRes(encounterModel.BossNodeSpineResource);
			_animController.GetAnimationState().AddAnimation("animation");
			_material = (ShaderMaterial)_animController.GetNormalMaterial();
		}
		else
		{
			_usesSpine = false;
			_spineSprite.Visible = false;
			_placeholderImage = GetNode<TextureRect>("%PlaceholderImage");
			_placeholderOutline = GetNode<TextureRect>("%PlaceholderOutline");
			_placeholderImage.Visible = true;
			_placeholderImage.Texture = PreloadManager.Cache.GetAsset<Texture2D>(encounterModel.BossNodePath + ".png");
			_placeholderOutline.Texture = PreloadManager.Cache.GetAsset<Texture2D>(encounterModel.BossNodePath + "_outline.png");
			_placeholderImage.SelfModulate = _act.MapTraveledColor;
			_placeholderOutline.SelfModulate = _act.MapBgColor;
		}
		RefreshColorInstantly();
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		if (IsInputAllowed() && base.IsTravelable)
		{
			_hoverTween?.Kill();
			_hoverTween = CreateTween().SetParallel();
			_hoverTween.TweenProperty(_spriteContainer, "scale", HoverScale, 0.05);
			_ = _usesSpine;
		}
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		if (IsInputAllowed())
		{
			_hoverTween?.Kill();
			_hoverTween = CreateTween().SetParallel();
			_hoverTween.TweenProperty(_spriteContainer, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
			_ = _usesSpine;
		}
	}

	protected override void OnPress()
	{
		if (base.IsTravelable)
		{
			_hoverTween?.Kill();
			_hoverTween = CreateTween().SetParallel();
			_hoverTween.TweenProperty(_spriteContainer, "scale", DownScale, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		}
	}

	public override void OnSelected()
	{
		base.State = MapPointState.Traveled;
	}

	protected override void RefreshColorInstantly()
	{
		if (_usesSpine)
		{
			MapPointState state = base.State;
			if ((uint)(state - 1) <= 1u)
			{
				_material.SetShaderParameter(_blackLayerColor, _act.MapTraveledColor);
			}
			else
			{
				_material.SetShaderParameter(_blackLayerColor, _act.MapUntraveledColor);
			}
			_material.SetShaderParameter(_mapColor, _act.MapBgColor);
		}
		else
		{
			MapPointState state = base.State;
			bool flag = (uint)(state - 1) <= 1u;
			Color selfModulate = (flag ? _act.MapTraveledColor : _act.MapUntraveledColor);
			_placeholderImage.SelfModulate = selfModulate;
			_placeholderOutline.SelfModulate = _act.MapBgColor;
		}
	}
}
