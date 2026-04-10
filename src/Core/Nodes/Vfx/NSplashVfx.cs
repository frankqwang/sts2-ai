using System.Threading;
using System.Threading.Tasks;
using Godot;
using Godot.Collections;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NSplashVfx : Node2D
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/vfx_splash");

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _particles = new Array<GpuParticles2D>();

	private Color _tint = Colors.White;

	private CancellationTokenSource? _cts;

	public static NSplashVfx? Create(Vector2 targetPosition, Color tint)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NSplashVfx nSplashVfx = PreloadManager.Cache.GetScene(scenePath).Instantiate<NSplashVfx>(PackedScene.GenEditState.Disabled);
		nSplashVfx.GlobalPosition = targetPosition;
		nSplashVfx._tint = tint;
		return nSplashVfx;
	}

	public override void _Ready()
	{
		TaskHelper.RunSafely(PlayVfx());
	}

	public override void _ExitTree()
	{
		_cts?.Cancel();
		_cts?.Dispose();
	}

	private async Task PlayVfx()
	{
		_cts = new CancellationTokenSource();
		foreach (GpuParticles2D particle in _particles)
		{
			particle.SelfModulate = _tint;
			particle.Restart();
		}
		await Cmd.Wait(2f, _cts.Token);
		this.QueueFreeSafely();
	}
}
