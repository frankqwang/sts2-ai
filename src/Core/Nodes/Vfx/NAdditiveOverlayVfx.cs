using System;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NAdditiveOverlayVfx : ColorRect
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("vfx/additive_overlay_vfx");

	private VfxColor _vfxColor;

	private Tween? _tween;

	public static NAdditiveOverlayVfx? Create(VfxColor vfxColor = VfxColor.Red)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NAdditiveOverlayVfx nAdditiveOverlayVfx = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NAdditiveOverlayVfx>(PackedScene.GenEditState.Disabled);
		nAdditiveOverlayVfx._vfxColor = vfxColor;
		return nAdditiveOverlayVfx;
	}

	public override void _Ready()
	{
		SetVfxColor();
		_tween = CreateTween();
		_tween.TweenProperty(this, "modulate:a", 0.1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenInterval(0.5);
		_tween.TweenProperty(this, "modulate:a", 0f, 0.5);
		_tween.Finished += OnTweenFinished;
	}

	private void SetVfxColor()
	{
		switch (_vfxColor)
		{
		case VfxColor.Green:
			base.Modulate = new Color("00ff1500");
			break;
		case VfxColor.Blue:
			base.Modulate = new Color("001aff00");
			break;
		case VfxColor.Purple:
			base.Modulate = new Color("b300ff00");
			break;
		case VfxColor.White:
			base.Modulate = new Color("ffffff00");
			break;
		case VfxColor.Cyan:
			base.Modulate = new Color("00fffb00");
			break;
		case VfxColor.Gold:
			base.Modulate = new Color("b17e0000");
			break;
		default:
			throw new ArgumentOutOfRangeException();
		case VfxColor.Red:
		case VfxColor.Black:
			break;
		}
	}

	private void OnTweenFinished()
	{
		this.QueueFreeSafely();
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
		if (_tween != null)
		{
			_tween.Finished -= OnTweenFinished;
		}
	}
}
