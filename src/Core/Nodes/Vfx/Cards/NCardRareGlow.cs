using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx.Cards;

public partial class NCardRareGlow : GpuParticles2D
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("vfx/rare_glow_vfx");

	private Tween? _tween;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public static NCardRareGlow? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NCardRareGlow>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		Vector2 vector = base.Scale * 0.92f;
		double delay = Rng.Chaotic.NextDouble(0.0, 0.2);
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "scale", vector, 1.0).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Cubic)
			.SetDelay(delay);
		_tween.TweenProperty(this, "modulate:a", 0.9f, 1.0).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Cubic)
			.SetDelay(delay);
	}

	public void Kill()
	{
		_tween?.Kill();
		_tween = CreateTween();
		_tween.TweenProperty(this, "modulate:a", 0, 0.25);
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
	}
}
