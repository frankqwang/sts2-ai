using System;
using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.CommonUi;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NTargetingArrow : Node2D
{
	private const int _segmentCount = 19;

	private Vector2 _fromPos;

	private Control? _fromControl;

	private Vector2 _toPosition;

	private Vector2? _currentArrowPos;

	private static readonly string _segmentHeadPath = ImageHelper.GetImagePath("ui/combat/targeting_arrow_head.png");

	private static readonly string _segmentBlockPath = ImageHelper.GetImagePath("ui/combat/targeting_arrow_segment.png");

	private Sprite2D[] _segments = new Sprite2D[19];

	private Sprite2D _arrowHead;

	private Tween? _arrowHeadTween;

	private bool _initialized;

	private bool _followMouse;

	private const float _segmentScaleStart = 0.28f;

	private const float _segmentScaleEnd = 0.42f;

	private static readonly Vector2 _arrowHeadDefaultScale = Vector2.One * 0.95f;

	private static readonly Vector2 _arrowHeadHoverScale = Vector2.One * 1.05f;

	private Vector2 From => _fromControl?.GlobalPosition ?? _fromPos;

	private static Texture2D SegmentHead => PreloadManager.Cache.GetTexture2D(_segmentHeadPath);

	private static Texture2D SegmentBlock => PreloadManager.Cache.GetTexture2D(_segmentBlockPath);

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlyArray<string>(new string[2] { _segmentHeadPath, _segmentBlockPath });

	public override void _Ready()
	{
		if (!_initialized)
		{
			_initialized = true;
			for (int i = 0; i < 19; i++)
			{
				_segments[i] = new Sprite2D();
				_segments[i].Texture = SegmentBlock;
				this.AddChildSafely(_segments[i]);
			}
			_arrowHead = new Sprite2D();
			_arrowHead.Texture = SegmentHead;
			this.AddChildSafely(_arrowHead);
			StopDrawing();
		}
	}

	public override void _Process(double delta)
	{
		if (base.Visible)
		{
			if (_followMouse)
			{
				UpdateDrawingTo(GetViewport().GetMousePosition());
				UpdateArrowPosition(_toPosition);
			}
			else
			{
				_currentArrowPos = _currentArrowPos.Value.Lerp(_toPosition, (float)delta * 14f);
				UpdateArrowPosition(_currentArrowPos.Value);
			}
		}
	}

	private void UpdateArrowPosition(Vector2 targetPos)
	{
		_arrowHead.Position = targetPos + new Vector2(0f, 88f).Rotated(_arrowHead.Rotation);
		Vector2 finalPos = targetPos + new Vector2(0f, 40f).Rotated(_arrowHead.Rotation);
		Vector2 zero = Vector2.Zero;
		zero.X = From.X - (_arrowHead.Position.X - From.X) * 0.25f;
		if (From.Y > 540f)
		{
			zero.Y = _arrowHead.Position.Y + (_arrowHead.Position.Y - From.Y) * 0.5f;
		}
		else
		{
			zero.Y = _arrowHead.Position.Y * 0.75f + From.Y * 0.25f;
		}
		_arrowHead.Rotation = (targetPos - zero).Angle() + (float)Math.PI / 2f;
		UpdateSegments(From, finalPos, zero);
	}

	public void SetHighlightingOn(bool isEnemy)
	{
		_arrowHeadTween?.Kill();
		_arrowHeadTween = CreateTween();
		_arrowHeadTween.TweenProperty(_arrowHead, "scale", _arrowHeadHoverScale, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Elastic);
		base.Modulate = (isEnemy ? StsColors.targetingArrowEnemy : StsColors.targetingArrowAlly);
	}

	public void SetHighlightingOff()
	{
		_arrowHeadTween?.Kill();
		_arrowHead.Scale = _arrowHeadDefaultScale;
		base.Modulate = Colors.White;
	}

	private void UpdateSegments(Vector2 initialPos, Vector2 finalPos, Vector2 controlPoint)
	{
		for (int i = 0; i < 19; i++)
		{
			_segments[i].Scale = Vector2.One * Mathf.Lerp(0.28f, 0.42f, (float)i * 2f / 19f);
			_segments[i].Position = MathHelper.BezierCurve(initialPos, finalPos, controlPoint, (float)i / 20f);
			if (i == 0)
			{
				_segments[i].Rotation = ((_segments[i].GlobalPosition.Y > 540f) ? 0f : ((float)Math.PI));
			}
			else
			{
				_segments[i].Rotation = (_segments[i].Position - _segments[i - 1].Position).Angle() + (float)Math.PI / 2f;
			}
		}
		_segments[0].Rotation = (_segments[0].Position - _segments[1].Position).Angle() - (float)Math.PI / 2f;
	}

	public void StartDrawingFrom(Vector2 from, bool usingController)
	{
		_followMouse = !usingController;
		if (!NControllerManager.Instance.IsUsingController)
		{
			Input.MouseMode = Input.MouseModeEnum.Hidden;
		}
		_fromPos = from;
		base.Visible = !NCombatUi.IsDebugHideTargetingUi;
	}

	public void StartDrawingFrom(Control control, bool usingController)
	{
		_followMouse = !usingController;
		base.ZIndex = control.ZIndex + 1;
		Input.MouseMode = Input.MouseModeEnum.Hidden;
		_fromControl = control;
		base.Visible = !NCombatUi.IsDebugHideTargetingUi;
	}

	public void StopDrawing()
	{
		if (!NControllerManager.Instance.IsUsingController)
		{
			Input.MouseMode = Input.MouseModeEnum.Visible;
		}
		_fromControl = null;
		_currentArrowPos = null;
		base.Visible = false;
		SetHighlightingOff();
	}

	public void UpdateDrawingTo(Vector2 position)
	{
		_toPosition = position;
		if (!_followMouse && !_currentArrowPos.HasValue)
		{
			_currentArrowPos = _toPosition;
		}
	}
}
