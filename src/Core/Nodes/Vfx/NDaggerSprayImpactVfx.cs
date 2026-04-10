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
using MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NDaggerSprayImpactVfx : Node2D
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/vfx_dagger_spray_impact");

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _particles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _modulateParticles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private float _impactDelay = 0.1f;

	private CancellationTokenSource? _cts;

	public static NDaggerSprayImpactVfx? Create(Creature creature, Color tint, bool goingRight)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(creature);
		if (nCreature != null)
		{
			return Create(nCreature.VfxSpawnPosition, tint, goingRight);
		}
		return null;
	}

	public static NDaggerSprayImpactVfx? Create(Vector2 targetCenter, Color tint, bool goingRight)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NDaggerSprayImpactVfx nDaggerSprayImpactVfx = PreloadManager.Cache.GetScene(scenePath).Instantiate<NDaggerSprayImpactVfx>(PackedScene.GenEditState.Disabled);
		nDaggerSprayImpactVfx.GlobalPosition = targetCenter;
		nDaggerSprayImpactVfx.ApplyTint(tint);
		nDaggerSprayImpactVfx.Scale = new Vector2(goingRight ? 1 : (-1), 1f);
		return nDaggerSprayImpactVfx;
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

	public void ApplyTint(Color tint)
	{
		for (int i = 0; i < _modulateParticles.Count; i++)
		{
			_modulateParticles[i].SelfModulate = tint;
		}
	}

	private async Task PlaySequence()
	{
		_cts = new CancellationTokenSource();
		await Cmd.Wait(_impactDelay, _cts.Token);
		NGame.Instance?.ScreenShake(ShakeStrength.Weak, ShakeDuration.Short);
		for (int i = 0; i < _particles.Count; i++)
		{
			_particles[i].Restart();
		}
		await Cmd.Wait(2f, _cts.Token);
		this.QueueFreeSafely();
	}
}
