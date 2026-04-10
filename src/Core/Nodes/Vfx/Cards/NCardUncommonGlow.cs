using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx.Cards;

public partial class NCardUncommonGlow : GpuParticles2D
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("vfx/uncommon_glow_vfx");

	private Tween? _tween;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public static NCardUncommonGlow? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NCardUncommonGlow>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		_tween = CreateTween();
		_tween.TweenProperty(this, "modulate:a", 0.9f, 1.0).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Cubic);
	}

	public void Kill()
	{
		_tween?.Kill();
		_tween = CreateTween();
		_tween.TweenProperty(this, "modulate:a", 0f, 0.25);
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
	}
}
