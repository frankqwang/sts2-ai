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

public partial class NScratchVfx : Node2D
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/vfx_scratch_impact");

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _anticipationParticles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _impactParticles = new Array<GpuParticles2D>();

	private CancellationTokenSource? _cts;

	public static NScratchVfx? Create(Creature creature, bool goingRight)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(creature);
		if (nCreature != null)
		{
			return Create(nCreature.VfxSpawnPosition, goingRight);
		}
		return null;
	}

	public static NScratchVfx? Create(Vector2 targetCenterPosition, bool goingRight)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NScratchVfx nScratchVfx = PreloadManager.Cache.GetScene(scenePath).Instantiate<NScratchVfx>(PackedScene.GenEditState.Disabled);
		nScratchVfx.GlobalPosition = targetCenterPosition;
		Vector2 one = Vector2.One;
		one.X *= (goingRight ? (-1f) : 1f);
		nScratchVfx.Scale = one;
		return nScratchVfx;
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
		for (int i = 0; i < _anticipationParticles.Count; i++)
		{
			_anticipationParticles[i].Restart();
		}
		await Cmd.Wait(0.1f, _cts.Token);
		for (int j = 0; j < _impactParticles.Count; j++)
		{
			_impactParticles[j].Restart();
		}
		NGame.Instance?.ScreenShake(ShakeStrength.VeryWeak, ShakeDuration.Short);
		await Cmd.Wait(2f, _cts.Token);
		this.QueueFreeSafely();
	}
}
