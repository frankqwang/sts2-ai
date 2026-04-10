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

public partial class NHyperbeamVfx : Node2D
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/vfx_hyperbeam");

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _anticipationParticles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _laserParticles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _laserEndParticles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Line2D? _laserLine;

	[Export(PropertyHint.None, "")]
	private Node2D? _laserContainer;

	public static readonly float hyperbeamAnticipationDuration = 0.525f;

	public static readonly float hyperbeamLaserDuration = 0.5f;

	public static NHyperbeamVfx? Create(Creature owner, Creature target)
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

	public static NHyperbeamVfx? Create(Vector2 defectEyePosition, Vector2 mainTargetCenterPosition)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NHyperbeamVfx nHyperbeamVfx = PreloadManager.Cache.GetScene(scenePath).Instantiate<NHyperbeamVfx>(PackedScene.GenEditState.Disabled);
		nHyperbeamVfx.GlobalPosition = defectEyePosition;
		nHyperbeamVfx.ApplyRotation(defectEyePosition, mainTargetCenterPosition);
		return nHyperbeamVfx;
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

	private void ShowLaser(bool showing)
	{
		for (int i = 0; i < _laserParticles.Count; i++)
		{
			_laserParticles[i].Visible = showing;
			if (showing)
			{
				_laserParticles[i].Restart();
			}
		}
		_laserLine.Visible = showing;
		_laserContainer.Visible = showing;
	}

	private async Task PlaySequence()
	{
		ShowLaser(showing: false);
		for (int i = 0; i < _anticipationParticles.Count; i++)
		{
			_anticipationParticles[i].Restart();
		}
		await Cmd.Wait(hyperbeamAnticipationDuration);
		ShowLaser(showing: true);
		NGame.Instance?.ScreenShake(ShakeStrength.Medium, ShakeDuration.Normal);
		await Cmd.Wait(hyperbeamLaserDuration);
		ShowLaser(showing: false);
		for (int j = 0; j < _laserEndParticles.Count; j++)
		{
			_laserEndParticles[j].Restart();
		}
		NGame.Instance?.ScreenShake(ShakeStrength.Strong, ShakeDuration.Short);
		await Cmd.Wait(2f);
		this.QueueFreeSafely();
	}
}
