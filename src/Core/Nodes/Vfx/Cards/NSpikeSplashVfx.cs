using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx.Cards;

public partial class NSpikeSplashVfx : Node2D
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("vfx/spike_splash_vfx");

	private float _duration = 1f;

	private int _spikeAmount = 6;

	private Vector2 _spawnPosition;

	private VfxColor _vfxColor;

	public static NSpikeSplashVfx? Create(Creature target, VfxColor vfxColor = VfxColor.Red)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NSpikeSplashVfx nSpikeSplashVfx = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NSpikeSplashVfx>(PackedScene.GenEditState.Disabled);
		nSpikeSplashVfx._spawnPosition = NCombatRoom.Instance.GetCreatureNode(target).GetBottomOfHitbox();
		nSpikeSplashVfx._vfxColor = vfxColor;
		return nSpikeSplashVfx;
	}

	public override void _Ready()
	{
		for (int i = 0; i < _spikeAmount; i++)
		{
			NFgGroundSpikeVfx child = NFgGroundSpikeVfx.Create(_spawnPosition, movingRight: true, _vfxColor);
			NCombatRoom.Instance.CombatVfxContainer.AddChildSafely(child);
			child = NFgGroundSpikeVfx.Create(_spawnPosition, movingRight: false, _vfxColor);
			NCombatRoom.Instance.CombatVfxContainer.AddChildSafely(child);
		}
		for (int j = 0; j < _spikeAmount; j++)
		{
			NBgGroundSpikeVfx child2 = NBgGroundSpikeVfx.Create(_spawnPosition, movingRight: true, _vfxColor);
			NCombatRoom.Instance.BackCombatVfxContainer.AddChildSafely(child2);
			child2 = NBgGroundSpikeVfx.Create(_spawnPosition, movingRight: false, _vfxColor);
			NCombatRoom.Instance.BackCombatVfxContainer.AddChildSafely(child2);
		}
		TaskHelper.RunSafely(SelfDestruct());
	}

	private async Task SelfDestruct()
	{
		await Task.Delay(2000);
		this.QueueFreeSafely();
	}
}
