using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NKinPriestGrenadeVfx : Node2D
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("vfx/monsters/kin_priest_grenade_vfx");

	private GpuParticles2D _cryptoParticles;

	private GpuParticles2D _noiseParticles;

	private GpuParticles2D _explosionBase;

	private readonly CancellationTokenSource _cancelToken = new CancellationTokenSource();

	public static NKinPriestGrenadeVfx? Create(Creature target)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCreature creatureNode = NCombatRoom.Instance.GetCreatureNode(target);
		if (creatureNode == null)
		{
			return null;
		}
		NKinPriestGrenadeVfx nKinPriestGrenadeVfx = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NKinPriestGrenadeVfx>(PackedScene.GenEditState.Disabled);
		nKinPriestGrenadeVfx.GlobalPosition = creatureNode.GetBottomOfHitbox();
		return nKinPriestGrenadeVfx;
	}

	public override void _Ready()
	{
		_cryptoParticles = GetNode<GpuParticles2D>("CryptoParticles");
		_noiseParticles = GetNode<GpuParticles2D>("NoiseParticles");
		_explosionBase = GetNode<GpuParticles2D>("ExplosionBaseParticle");
		_cryptoParticles.Emitting = false;
		_cryptoParticles.OneShot = true;
		_noiseParticles.Emitting = false;
		_noiseParticles.OneShot = true;
		_explosionBase.Emitting = false;
		_explosionBase.OneShot = true;
		TaskHelper.RunSafely(Play());
	}

	private async Task Play()
	{
		NDebugAudioManager.Instance?.Play("blunt_attack.mp3");
		_noiseParticles.SetEmitting(emitting: true);
		_explosionBase.SetEmitting(emitting: true);
		await Task.Delay(100, _cancelToken.Token);
		_cryptoParticles.SetEmitting(emitting: true);
		await Task.Delay(5000, _cancelToken.Token);
	}
}
