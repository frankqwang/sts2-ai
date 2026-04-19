using System.Threading.Tasks;
using Godot;
using Godot.Collections;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NPowerAppliedDebuffVfx : Node2D
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/ui/vfx_debuff_applied");

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _particles = new Array<GpuParticles2D>();

	public static NPowerAppliedDebuffVfx? Create(Vector2 globalPosition)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NPowerAppliedDebuffVfx nPowerAppliedDebuffVfx = PreloadManager.Cache.GetScene(scenePath).Instantiate<NPowerAppliedDebuffVfx>(PackedScene.GenEditState.Disabled);
		nPowerAppliedDebuffVfx.GlobalPosition = globalPosition;
		return nPowerAppliedDebuffVfx;
	}

	public override void _Ready()
	{
		TaskHelper.RunSafely(PlaySequence());
	}

	private async Task PlaySequence()
	{
		for (int i = 0; i < _particles.Count; i++)
		{
			_particles[i].Restart();
		}
		await Cmd.Wait(2f);
		this.QueueFreeSafely();
	}
}
