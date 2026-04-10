using System.Threading.Tasks;
using Godot;
using Godot.Collections;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NSweepingBeamImpactVfx : Node2D
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/vfx_sweeping_beam_impact");

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _impactParticles = new Array<GpuParticles2D>();

	public static NSweepingBeamImpactVfx? Create(Creature creature)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(creature);
		if (nCreature != null)
		{
			return Create(nCreature.VfxSpawnPosition);
		}
		return null;
	}

	public static NSweepingBeamImpactVfx? Create(Vector2 targetCenter)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NSweepingBeamImpactVfx nSweepingBeamImpactVfx = PreloadManager.Cache.GetScene(scenePath).Instantiate<NSweepingBeamImpactVfx>(PackedScene.GenEditState.Disabled);
		nSweepingBeamImpactVfx.GlobalPosition = targetCenter;
		return nSweepingBeamImpactVfx;
	}

	public override void _Ready()
	{
		TaskHelper.RunSafely(PlaySequence());
	}

	private async Task PlaySequence()
	{
		for (int i = 0; i < _impactParticles.Count; i++)
		{
			_impactParticles[i].Restart();
		}
		await Cmd.Wait(2f);
		this.QueueFreeSafely();
	}
}
