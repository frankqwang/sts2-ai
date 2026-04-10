using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NDesaturateTransitionVfx : Node
{
	private Tween? _tween;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	private static string ScenePath => SceneHelper.GetScenePath("vfx/desaturate_transition_vfx");

	public static NDesaturateTransitionVfx? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(ScenePath).Instantiate<NDesaturateTransitionVfx>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		Animate();
	}

	public override void _ExitTree()
	{
		if (_tween != null && _tween.IsRunning())
		{
			_tween.Kill();
			NGame.Instance.DeactivateWorldEnvironment();
		}
	}

	private void Animate()
	{
		WorldEnvironment worldEnvironment = NGame.Instance.ActivateWorldEnvironment();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(worldEnvironment, "environment:tonemap_exposure", 1f, 1.0);
		_tween.TweenProperty(worldEnvironment, "environment:adjustment_brightness", 0.1f, 1.0);
		_tween.TweenProperty(worldEnvironment, "environment:adjustment_contrast", 0.7f, 1.0);
		_tween.TweenProperty(worldEnvironment, "environment:adjustment_saturation", 0.25f, 1.0);
		_tween.Chain();
		_tween.TweenProperty(worldEnvironment, "environment:adjustment_contrast", 0.8f, 1.0);
		_tween.Chain();
		_tween.TweenProperty(worldEnvironment, "environment:adjustment_brightness", 1f, 1.0);
		_tween.TweenProperty(worldEnvironment, "environment:adjustment_contrast", 1f, 1.0);
		_tween.TweenProperty(worldEnvironment, "environment:adjustment_saturation", 1f, 1.0);
		_tween.Chain().TweenCallback(Callable.From(delegate
		{
			NGame.Instance.DeactivateWorldEnvironment();
			this.QueueFreeSafely();
		}));
	}
}
