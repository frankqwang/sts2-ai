using System.Threading;
using System.Threading.Tasks;
using Godot;
using Godot.Collections;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NSpookyScreamVfx : Node2D
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/vfx_spooky_scream");

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _continuousParticles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _oneShotParticles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private float _duration = 1f;

	private CancellationTokenSource? _cts;

	public static NSpookyScreamVfx? Create(Vector2 position)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NSpookyScreamVfx nSpookyScreamVfx = PreloadManager.Cache.GetScene(scenePath).Instantiate<NSpookyScreamVfx>(PackedScene.GenEditState.Disabled);
		nSpookyScreamVfx.GlobalPosition = position;
		return nSpookyScreamVfx;
	}

	public override void _Ready()
	{
		TaskHelper.RunSafely(PlaySequence());
	}

	public override void _ExitTree()
	{
		_cts?.Cancel();
		_cts?.Dispose();
	}

	private async Task PlaySequence()
	{
		_cts = new CancellationTokenSource();
		for (int i = 0; i < _oneShotParticles.Count; i++)
		{
			_oneShotParticles[i].Lifetime = _duration;
			_oneShotParticles[i].Restart();
		}
		for (int j = 0; j < _continuousParticles.Count; j++)
		{
			_continuousParticles[j].Restart();
			_continuousParticles[j].Emitting = true;
		}
		await Cmd.Wait(_duration, _cts.Token);
		for (int k = 0; k < _continuousParticles.Count; k++)
		{
			_continuousParticles[k].Emitting = false;
		}
		await Cmd.Wait(2f, _cts.Token);
		this.QueueFreeSafely();
	}
}
