using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using Godot.Collections;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NBlockSparkVfx : Node2D
{
	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _particles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private GpuParticles2D _specks;

	private NCreature _creatureNode;

	private static string ScenePath => SceneHelper.GetScenePath("vfx/block_spark_vfx");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	public static NBlockSparkVfx? Create(Creature target)
	{
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(target);
		if (nCreature == null || !nCreature.IsInteractable)
		{
			return null;
		}
		NBlockSparkVfx nBlockSparkVfx = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NBlockSparkVfx>(PackedScene.GenEditState.Disabled);
		nBlockSparkVfx._creatureNode = nCreature;
		return nBlockSparkVfx;
	}

	public override void _Ready()
	{
		base.GlobalPosition = _creatureNode.VfxSpawnPosition;
		for (int i = 0; i < _particles.Count; i++)
		{
			_particles[i].Restart();
		}
		TaskHelper.RunSafely(FlashAndFree());
	}

	private async Task FlashAndFree()
	{
		await ToSignal(_specks, GpuParticles2D.SignalName.Finished);
		this.QueueFreeSafely();
	}
}
