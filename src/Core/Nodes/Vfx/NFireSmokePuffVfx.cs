using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NFireSmokePuffVfx : Node2D
{
	private GpuParticles2D _clouds;

	private GpuParticles2D _ember;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	private static string ScenePath => SceneHelper.GetScenePath("vfx/vfx_fire_smoke_puff");

	public static NFireSmokePuffVfx? Create(Creature target)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NFireSmokePuffVfx nFireSmokePuffVfx = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NFireSmokePuffVfx>(PackedScene.GenEditState.Disabled);
		nFireSmokePuffVfx.GlobalPosition = NCombatRoom.Instance.GetCreatureNode(target).VfxSpawnPosition;
		return nFireSmokePuffVfx;
	}

	public override void _Ready()
	{
		_ember = GetNode<GpuParticles2D>("Ember");
		_clouds = GetNode<GpuParticles2D>("Clouds");
		_ember.Emitting = true;
		_clouds.Emitting = true;
		TaskHelper.RunSafely(DeleteAfterComplete());
	}

	private async Task DeleteAfterComplete()
	{
		await Task.Delay(2500);
		this.QueueFreeSafely();
	}
}
