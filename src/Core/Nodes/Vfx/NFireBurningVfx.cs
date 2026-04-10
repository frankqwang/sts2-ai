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

public partial class NFireBurningVfx : Node2D
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/vfx_fire_burning");

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _startParticles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _endParticles = new Array<GpuParticles2D>();

	private CancellationTokenSource? _cts;

	private static Color DefaultColor => Color.FromHtml("#ff8b57");

	public static NFireBurningVfx? Create(Creature creature, float scaleFactor, bool goingRight)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(creature);
		if (nCreature != null)
		{
			return Create(nCreature.GetBottomOfHitbox(), scaleFactor, goingRight);
		}
		return null;
	}

	public static NFireBurningVfx? Create(Vector2 targetFloorPosition, float scaleFactor, bool goingRight)
	{
		return Create(targetFloorPosition, scaleFactor, goingRight, DefaultColor);
	}

	public static NFireBurningVfx? Create(Vector2 targetFloorPosition, float scaleFactor, bool goingRight, Color tint)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NFireBurningVfx nFireBurningVfx = PreloadManager.Cache.GetScene(scenePath).Instantiate<NFireBurningVfx>(PackedScene.GenEditState.Disabled);
		nFireBurningVfx.GlobalPosition = targetFloorPosition;
		nFireBurningVfx.Modulate = tint;
		Vector2 scale = Vector2.One * scaleFactor;
		scale.X *= (goingRight ? 1f : (-1f));
		nFireBurningVfx.Scale = scale;
		return nFireBurningVfx;
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
		for (int i = 0; i < _startParticles.Count; i++)
		{
			_startParticles[i].Restart();
		}
		await Cmd.Wait(0.3f, _cts.Token);
		for (int j = 0; j < _endParticles.Count; j++)
		{
			_endParticles[j].Restart();
		}
		await Cmd.Wait(2f, _cts.Token);
		this.QueueFreeSafely();
	}
}
