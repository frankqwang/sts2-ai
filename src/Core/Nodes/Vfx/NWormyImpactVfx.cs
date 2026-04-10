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

public partial class NWormyImpactVfx : Node2D
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/vfx_wormy_impact");

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _particles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Node2D? _centerPivot;

	[Export(PropertyHint.None, "")]
	private Node2D? _groundPivot;

	public static NWormyImpactVfx? Create(Creature creature)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(creature);
		if (nCreature != null)
		{
			return Create(nCreature.GetBottomOfHitbox(), nCreature.VfxSpawnPosition);
		}
		return null;
	}

	public static NWormyImpactVfx? Create(Vector2 targetGroundPosition, Vector2 targetCenterPosition)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NWormyImpactVfx nWormyImpactVfx = PreloadManager.Cache.GetScene(scenePath).Instantiate<NWormyImpactVfx>(PackedScene.GenEditState.Disabled);
		nWormyImpactVfx.GlobalPosition = targetGroundPosition;
		nWormyImpactVfx.Initialize(targetGroundPosition, targetCenterPosition);
		return nWormyImpactVfx;
	}

	private void Initialize(Vector2 targetGroundPosition, Vector2 targetCenterPosition)
	{
		_groundPivot.GlobalPosition = targetGroundPosition;
		_centerPivot.GlobalPosition = targetCenterPosition;
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
