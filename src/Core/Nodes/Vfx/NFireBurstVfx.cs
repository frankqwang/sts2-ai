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

public partial class NFireBurstVfx : Node2D
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/vfx_fire_burst");

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _particles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _modulateParticles = new Array<GpuParticles2D>();

	private CancellationTokenSource? _cts;

	private static Color DefaultColor => Color.FromHtml("#ff8b57");

	public static NFireBurstVfx? Create(Creature creature, float scaleFactor)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(creature);
		if (nCreature != null)
		{
			return Create(nCreature.GetBottomOfHitbox(), scaleFactor, DefaultColor);
		}
		return null;
	}

	public static NFireBurstVfx? Create(Vector2 targetFloorPosition, float scaleFactor)
	{
		return Create(targetFloorPosition, scaleFactor, DefaultColor);
	}

	public static NFireBurstVfx? Create(Vector2 targetFloorPosition, float scaleFactor, Color tint)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NFireBurstVfx nFireBurstVfx = PreloadManager.Cache.GetScene(scenePath).Instantiate<NFireBurstVfx>(PackedScene.GenEditState.Disabled);
		nFireBurstVfx.GlobalPosition = targetFloorPosition;
		nFireBurstVfx.ApplyTint(tint);
		Vector2 scale = Vector2.One * scaleFactor;
		scale.X *= (((double)GD.Randf() > 0.5) ? 1f : (-1f));
		nFireBurstVfx.Scale = scale;
		return nFireBurstVfx;
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
		for (int i = 0; i < _particles.Count; i++)
		{
			_particles[i].Restart();
		}
		NGame.Instance?.ScreenShake(ShakeStrength.Weak, ShakeDuration.Normal);
		await Cmd.Wait(2f, _cts.Token);
		this.QueueFreeSafely();
	}
}
