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

public partial class NBigSlashImpactVfx : Node2D
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/vfx_big_slash_impact");

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _anticipationParticles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _impactParticles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _modulateParticles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Node2D? _corePivot;

	private CancellationTokenSource? _cts;

	public static NBigSlashImpactVfx? Create(Creature creature)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(creature);
		if (nCreature != null)
		{
			return Create(nCreature.VfxSpawnPosition);
		}
		return null;
	}

	public static NBigSlashImpactVfx? Create(Vector2 targetCenterPosition)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return Create(targetCenterPosition, 60f, Color.FromHtml("#80dbff"));
	}

	public static NBigSlashImpactVfx? Create(Vector2 targetCenterPosition, float rotationDegrees, Color tint)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NBigSlashImpactVfx nBigSlashImpactVfx = PreloadManager.Cache.GetScene(scenePath).Instantiate<NBigSlashImpactVfx>(PackedScene.GenEditState.Disabled);
		nBigSlashImpactVfx.GlobalPosition = targetCenterPosition;
		nBigSlashImpactVfx.RotateCore(rotationDegrees);
		nBigSlashImpactVfx.ModulateParticles(tint);
		return nBigSlashImpactVfx;
	}

	public override void _Ready()
	{
		TaskHelper.RunSafely(PlaySequence());
	}

	private void RotateCore(float rotationDegrees)
	{
		if (_corePivot != null)
		{
			_corePivot.RotationDegrees = rotationDegrees;
		}
	}

	private void ModulateParticles(Color tint)
	{
		for (int i = 0; i < _modulateParticles.Count; i++)
		{
			_modulateParticles[i].SelfModulate = tint;
		}
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
		NGame.Instance?.ScreenShake(ShakeStrength.Strong, ShakeDuration.Short);
		for (int j = 0; j < _impactParticles.Count; j++)
		{
			_impactParticles[j].Restart();
		}
		await Cmd.Wait(2f, _cts.Token);
		this.QueueFreeSafely();
	}
}
