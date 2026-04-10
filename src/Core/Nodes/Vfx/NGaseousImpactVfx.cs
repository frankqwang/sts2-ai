using System.Threading;
using System.Threading.Tasks;
using Godot;
using Godot.Collections;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NGaseousImpactVfx : Node2D
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/vfx_gaseous_impact");

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _impactParticles = new Array<GpuParticles2D>();

	private CancellationTokenSource? _cts;

	public static NGaseousImpactVfx? Create(CombatSide side, CombatState combatState, Color tint)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return Create(VfxCmd.GetSideCenter(side, combatState).Value, tint);
	}

	public static NGaseousImpactVfx? Create(Creature creature, Color tint)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(creature);
		if (nCreature != null)
		{
			return Create(nCreature.VfxSpawnPosition, tint);
		}
		return null;
	}

	public static NGaseousImpactVfx? Create(Vector2 targetCenter, Color tint)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NGaseousImpactVfx nGaseousImpactVfx = PreloadManager.Cache.GetScene(scenePath).Instantiate<NGaseousImpactVfx>(PackedScene.GenEditState.Disabled);
		nGaseousImpactVfx.GlobalPosition = targetCenter;
		nGaseousImpactVfx.Modulate = tint;
		return nGaseousImpactVfx;
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
		for (int i = 0; i < _impactParticles.Count; i++)
		{
			_impactParticles[i].Restart();
		}
		await Cmd.Wait(2f, _cts.Token);
		this.QueueFreeSafely();
	}
}
