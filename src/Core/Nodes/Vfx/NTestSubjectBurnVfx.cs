using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NTestSubjectBurnVfx : ColorRect
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("vfx/monsters/test_subject_burn_vfx");

	public static NTestSubjectBurnVfx? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NTestSubjectBurnVfx>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		base._Ready();
		base.Modulate = Colors.Transparent;
		Tween tween = CreateTween();
		tween.Chain().TweenInterval(0.25);
		tween.Chain().TweenProperty(this, "modulate", Colors.White, 0.30000001192092896);
		tween.Chain().TweenInterval(0.10000000149011612);
		tween.Chain().TweenProperty(this, "modulate", Colors.White * 0.5f, 0.15000000596046448);
		tween.Chain().TweenProperty(this, "modulate", Colors.White, 0.25);
		tween.Chain().TweenInterval(0.3499999940395355);
		tween.Chain().TweenProperty(this, "modulate", Colors.Transparent, 0.30000001192092896);
		tween.Chain().TweenCallback(Callable.From(this.QueueFreeSafely));
	}
}
