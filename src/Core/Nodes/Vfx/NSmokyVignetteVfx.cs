using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NSmokyVignetteVfx : Control
{
	private const string _path = "res://scenes/vfx/whole_screen/vfx_smoky_vignette.tscn";

	private Control _highlights;

	private Color _targetColor;

	private Color _highlightColor;

	private Tween? _tween;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>("res://scenes/vfx/whole_screen/vfx_smoky_vignette.tscn");

	public static NSmokyVignetteVfx? Create(Color tint, Color highlightColor)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NSmokyVignetteVfx nSmokyVignetteVfx = PreloadManager.Cache.GetScene("res://scenes/vfx/whole_screen/vfx_smoky_vignette.tscn").Instantiate<NSmokyVignetteVfx>(PackedScene.GenEditState.Disabled);
		nSmokyVignetteVfx.Modulate = tint;
		nSmokyVignetteVfx._targetColor = tint;
		nSmokyVignetteVfx._highlightColor = highlightColor;
		return nSmokyVignetteVfx;
	}

	public override void _Ready()
	{
		_highlights = GetNode<Control>("Highlights");
		_highlights.Modulate = _highlightColor;
		TaskHelper.RunSafely(Animate(fadeIn: true));
	}

	public void Reset(Color tint, Color highlightColor)
	{
		_targetColor = tint;
		_highlightColor = highlightColor;
		_tween?.Kill();
		TaskHelper.RunSafely(Animate(fadeIn: false));
	}

	private async Task Animate(bool fadeIn)
	{
		_tween = CreateTween().SetParallel();
		if (fadeIn)
		{
			_tween.TweenProperty(this, "modulate:a", _targetColor.A, 0.1).From(0f);
			_tween.TweenProperty(_highlights, "modulate:a", _highlightColor.A, 0.1).From(0f);
		}
		_tween.Chain();
		_tween.TweenProperty(this, "modulate:a", 0f, 1.0).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Cubic);
		_tween.TweenProperty(_highlights, "modulate:a", 0f, 1.0);
		while (_tween.IsValid() && _tween.IsRunning())
		{
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		}
		if (_tween.IsValid())
		{
			this.QueueFreeSafely();
		}
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
	}
}
