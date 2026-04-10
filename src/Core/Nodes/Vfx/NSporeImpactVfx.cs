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

public partial class NSporeImpactVfx : Node2D
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/vfx_spore_impact");

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _impactParticles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Node2D? _poofPivot;

	[Export(PropertyHint.None, "")]
	private Vector2 _scaleRange;

	private CancellationTokenSource? _cts;

	public static NSporeImpactVfx? Create(Creature creature, Color color)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(creature);
		if (nCreature != null)
		{
			return Create(nCreature.GetBottomOfHitbox(), color);
		}
		return null;
	}

	public static NSporeImpactVfx? Create(Vector2 targetGroundPosition, Color tint)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NSporeImpactVfx nSporeImpactVfx = PreloadManager.Cache.GetScene(scenePath).Instantiate<NSporeImpactVfx>(PackedScene.GenEditState.Disabled);
		nSporeImpactVfx.GlobalPosition = targetGroundPosition;
		nSporeImpactVfx.Initialize();
		nSporeImpactVfx.ModulateParticles(tint);
		return nSporeImpactVfx;
	}

	private void Initialize()
	{
		_poofPivot.Scale = Vector2.One * (float)GD.RandRange(_scaleRange.X, _scaleRange.Y);
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
