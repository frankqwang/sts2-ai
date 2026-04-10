using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using Godot.Collections;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models.Characters;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NSweepingBeamVfx : Node2D
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/vfx_sweeping_beam");

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _emittingParticles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _startParticles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _endParticles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _sweepingParticles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Curve? _sweepingIndexCurve;

	[Export(PropertyHint.None, "")]
	private float _sweepDuration = 0.65f;

	private Array<Vector2> _targetCenterPositions = new Array<Vector2>();

	public static NSweepingBeamVfx? Create(Creature owner, List<Creature> targets)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(owner);
		if (nCreature != null)
		{
			Vector2 vfxSpawnPosition = nCreature.VfxSpawnPosition;
			Player player = owner.Player;
			if (player != null && player.Character is Defect defect)
			{
				vfxSpawnPosition += defect.EyelineOffset;
			}
			Array<Vector2> array = new Array<Vector2>();
			foreach (Creature target in targets)
			{
				NCreature nCreature2 = NCombatRoom.Instance?.GetCreatureNode(target);
				if (nCreature2 != null)
				{
					array.Add(nCreature2.VfxSpawnPosition);
				}
			}
			return Create(vfxSpawnPosition, array);
		}
		return null;
	}

	public static NSweepingBeamVfx? Create(Vector2 defectEyeCenter, Array<Vector2> targetCenterPositions)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NSweepingBeamVfx nSweepingBeamVfx = PreloadManager.Cache.GetScene(scenePath).Instantiate<NSweepingBeamVfx>(PackedScene.GenEditState.Disabled);
		nSweepingBeamVfx.GlobalPosition = defectEyeCenter;
		nSweepingBeamVfx._targetCenterPositions = targetCenterPositions;
		return nSweepingBeamVfx;
	}

	public override void _Ready()
	{
		for (int i = 0; i < _emittingParticles.Count; i++)
		{
			_emittingParticles[i].Emitting = false;
		}
		TaskHelper.RunSafely(PlaySequence());
	}

	private async Task PlaySequence()
	{
		double timer = 0.0;
		bool playedImpactParticles = false;
		for (int i = 0; i < _startParticles.Count; i++)
		{
			_startParticles[i].Restart();
		}
		for (int j = 0; j < _emittingParticles.Count; j++)
		{
			_emittingParticles[j].Restart();
			_emittingParticles[j].Emitting = true;
		}
		int previousSweepIndex = -1;
		while (timer < (double)_sweepDuration)
		{
			double processDeltaTime = GetProcessDeltaTime();
			float num = (float)(timer / (double)_sweepDuration);
			int num2 = Mathf.FloorToInt(_sweepingIndexCurve.Sample(num));
			if (previousSweepIndex != num2 && num2 >= 0 && num2 < _sweepingParticles.Count)
			{
				_sweepingParticles[num2].Restart();
				previousSweepIndex = num2;
			}
			if (num >= 0.5f && !playedImpactParticles)
			{
				playedImpactParticles = true;
				NGame.Instance?.ScreenShake(ShakeStrength.Medium, ShakeDuration.Normal);
				for (int k = 0; k < _targetCenterPositions.Count; k++)
				{
					NSweepingBeamImpactVfx child = NSweepingBeamImpactVfx.Create(_targetCenterPositions[k]);
					GetTree().Root.AddChildSafely(child);
				}
			}
			timer += processDeltaTime;
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		}
		for (int l = 0; l < _endParticles.Count; l++)
		{
			_endParticles[l].Restart();
		}
		for (int m = 0; m < _emittingParticles.Count; m++)
		{
			_emittingParticles[m].Emitting = false;
		}
		await Cmd.Wait(2f);
		this.QueueFreeSafely();
	}
}
