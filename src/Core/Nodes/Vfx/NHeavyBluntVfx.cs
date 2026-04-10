using System.Threading.Tasks;
using Godot;
using Godot.Collections;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NHeavyBluntVfx : Node2D
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/vfx_heavy_blunt");

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _anticipationParticles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _impactParticles = new Array<GpuParticles2D>();

	private Vector2 _debugPosition;

	public static NHeavyBluntVfx? Create(Vector2 debugPosition)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NHeavyBluntVfx nHeavyBluntVfx = PreloadManager.Cache.GetScene(scenePath).Instantiate<NHeavyBluntVfx>(PackedScene.GenEditState.Disabled);
		nHeavyBluntVfx._debugPosition = debugPosition;
		return nHeavyBluntVfx;
	}

	public override void _Ready()
	{
		TaskHelper.RunSafely(PlaySequence());
	}

	private async Task PlaySequence()
	{
		base.GlobalPosition = _debugPosition;
		for (int i = 0; i < _anticipationParticles.Count; i++)
		{
			_anticipationParticles[i].Restart();
		}
		await WaitForSeconds(0.2f);
		NGame.Instance?.ScreenShake(ShakeStrength.Strong, ShakeDuration.Short);
		for (int j = 0; j < _impactParticles.Count; j++)
		{
			_impactParticles[j].Restart();
		}
		await WaitForSeconds(2f);
		this.QueueFreeSafely();
	}

	private async Task WaitForSeconds(float duration)
	{
		double timer = 0.0;
		while (timer < (double)duration)
		{
			timer += GetProcessDeltaTime();
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		}
	}
}
