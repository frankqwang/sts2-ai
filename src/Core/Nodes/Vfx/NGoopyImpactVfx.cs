using System.Threading;
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

public partial class NGoopyImpactVfx : Node2D
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/vfx_goopy_impact");

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _impactParticles = new Array<GpuParticles2D>();

	private CancellationTokenSource? _cts;

	public static NGoopyImpactVfx? Create(Creature creature)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(creature);
		if (nCreature != null)
		{
			return Create(nCreature.VfxSpawnPosition, Colors.Green);
		}
		return null;
	}

	public static NGoopyImpactVfx? Create(Vector2 targetCenterPosition, Color tint)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NGoopyImpactVfx nGoopyImpactVfx = PreloadManager.Cache.GetScene(scenePath).Instantiate<NGoopyImpactVfx>(PackedScene.GenEditState.Disabled);
		nGoopyImpactVfx.GlobalPosition = targetCenterPosition;
		nGoopyImpactVfx.ModulateParticles(tint);
		return nGoopyImpactVfx;
	}

	private void ModulateParticles(Color tint)
	{
		base.Modulate = tint;
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
		await Cmd.Wait(3.5f, _cts.Token);
		this.QueueFreeSafely();
	}
}
