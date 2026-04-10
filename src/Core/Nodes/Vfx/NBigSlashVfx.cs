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

public partial class NBigSlashVfx : Node2D
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/vfx_big_slash");

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _slashParticles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _modulateParticles = new Array<GpuParticles2D>();

	private CancellationTokenSource? _cts;

	public static NBigSlashVfx? Create(Creature creature)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(creature);
		if (nCreature != null)
		{
			return Create(nCreature.VfxSpawnPosition, creature.IsEnemy, new Color("50b598"));
		}
		return null;
	}

	public static NBigSlashVfx? Create(Vector2 targetCenterPosition, bool facingRight)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return Create(targetCenterPosition, facingRight, Color.FromHtml("#a380ff"));
	}

	public static NBigSlashVfx? Create(Vector2 targetCenterPosition, bool facingRight, Color tint)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NBigSlashVfx nBigSlashVfx = PreloadManager.Cache.GetScene(scenePath).Instantiate<NBigSlashVfx>(PackedScene.GenEditState.Disabled);
		nBigSlashVfx.GlobalPosition = targetCenterPosition;
		nBigSlashVfx.Scale = new Vector2(facingRight ? 1f : (-1f), 1f);
		nBigSlashVfx.ModulateParticles(tint);
		return nBigSlashVfx;
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

	private void ModulateParticles(Color tint)
	{
		for (int i = 0; i < _modulateParticles.Count; i++)
		{
			_modulateParticles[i].SelfModulate = tint;
		}
	}

	private async Task PlaySequence()
	{
		_cts = new CancellationTokenSource();
		for (int i = 0; i < _slashParticles.Count; i++)
		{
			_slashParticles[i].Restart();
		}
		await Cmd.Wait(1f, _cts.Token);
		this.QueueFreeSafely();
	}
}
