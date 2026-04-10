using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NDoomOverlayVfx : BackBufferCopy
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("vfx/doom_overlay_vfx");

	private static NDoomOverlayVfx? _instance;

	private Tween? _tween;

	public static NDoomOverlayVfx? GetOrCreate()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		if (_instance != null)
		{
			_instance.PlayVfx();
		}
		else
		{
			_instance = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NDoomOverlayVfx>(PackedScene.GenEditState.Disabled);
		}
		if (!GodotObject.IsInstanceValid(_instance))
		{
			return null;
		}
		return _instance;
	}

	public override void _Ready()
	{
		base.Modulate = Colors.Transparent;
		GetNode<Control>("%Rect").SetDeferred(Control.PropertyName.Size, GetViewportRect().Size);
		PlayVfx();
	}

	private void PlayVfx()
	{
		if (GodotObject.IsInstanceValid(this))
		{
			_tween?.Kill();
			_tween = CreateTween();
			_tween.TweenProperty(this, "modulate:a", 1f, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			_tween.TweenInterval(1.2000000476837158);
			_tween.TweenProperty(this, "modulate:a", 0f, 0.5);
			_tween.TweenCallback(Callable.From(delegate
			{
				_instance = null;
			}));
			_tween.Finished += OnTweenFinished;
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
