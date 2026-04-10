using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NMapPingVfx : Control
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("vfx/map_ping_vfx");

	private Tween? _tween;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public static NMapPingVfx? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NMapPingVfx>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		TaskHelper.RunSafely(PlayAnim());
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
	}

	private async Task PlayAnim()
	{
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "scale", Vector2.One * 2.5f, 0.699999988079071).SetTrans(Tween.TransitionType.Expo).SetEase(Tween.EaseType.Out);
		Tween? tween = _tween;
		NMapPingVfx nMapPingVfx = this;
		NodePath property = "modulate";
		Color modulate = base.Modulate;
		modulate.A = 0f;
		tween.TweenProperty(nMapPingVfx, property, modulate, 0.15000000596046448).SetDelay(0.15000000596046448);
		await ToSignal(_tween, Tween.SignalName.Finished);
		this.QueueFreeSafely();
	}
}
