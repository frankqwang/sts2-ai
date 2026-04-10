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
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NHyperbeamImpactVfx : Node2D
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/vfx_hyperbeam_impact");

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _impactStartParticles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _impactEndParticles = new Array<GpuParticles2D>();

	public static NHyperbeamImpactVfx? Create(Creature owner, Creature target)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(owner);
		NCreature nCreature2 = NCombatRoom.Instance?.GetCreatureNode(target);
		if (nCreature2 != null && nCreature != null)
		{
			Vector2 vfxSpawnPosition = nCreature.VfxSpawnPosition;
			Player player = owner.Player;
			if (player != null && player.Character is Defect defect)
			{
				vfxSpawnPosition += defect.EyelineOffset;
			}
			return Create(vfxSpawnPosition, nCreature2.VfxSpawnPosition);
		}
		return null;
	}

	public static NHyperbeamImpactVfx? Create(Vector2 hyperbeamSourcePosition, Vector2 targetCenterPosition)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NHyperbeamImpactVfx nHyperbeamImpactVfx = PreloadManager.Cache.GetScene(scenePath).Instantiate<NHyperbeamImpactVfx>(PackedScene.GenEditState.Disabled);
		nHyperbeamImpactVfx.GlobalPosition = targetCenterPosition;
		nHyperbeamImpactVfx.ApplyRotation(hyperbeamSourcePosition, targetCenterPosition);
		return nHyperbeamImpactVfx;
	}

	public void ApplyRotation(Vector2 sourcePosition, Vector2 targetPosition)
	{
		Vector2 vector = targetPosition - sourcePosition;
		float rotationDegrees = Mathf.RadToDeg(Mathf.Atan2(vector.Y, vector.X));
		base.RotationDegrees = rotationDegrees;
	}

	public override void _Ready()
	{
		TaskHelper.RunSafely(PlaySequence());
	}

	private async Task PlaySequence()
	{
		for (int i = 0; i < _impactStartParticles.Count; i++)
		{
			_impactStartParticles[i].Visible = true;
			_impactStartParticles[i].Restart();
		}
		await Cmd.Wait(NHyperbeamVfx.hyperbeamLaserDuration);
		for (int j = 0; j < _impactStartParticles.Count; j++)
		{
			_impactStartParticles[j].Visible = false;
		}
		for (int k = 0; k < _impactEndParticles.Count; k++)
		{
			_impactEndParticles[k].Restart();
		}
		await Cmd.Wait(2f);
		this.QueueFreeSafely();
	}
}
