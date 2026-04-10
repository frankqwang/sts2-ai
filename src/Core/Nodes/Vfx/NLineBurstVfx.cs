using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NLineBurstVfx : GpuParticles2D
{
	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	private static string ScenePath => SceneHelper.GetScenePath("vfx/vfx_line_burst");

	public static NLineBurstVfx? Create(Creature target)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NLineBurstVfx nLineBurstVfx = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NLineBurstVfx>(PackedScene.GenEditState.Disabled);
		nLineBurstVfx.GlobalPosition = NCombatRoom.Instance.GetCreatureNode(target).VfxSpawnPosition;
		return nLineBurstVfx;
	}

	public static NLineBurstVfx? Create(Vector2 position)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NLineBurstVfx nLineBurstVfx = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NLineBurstVfx>(PackedScene.GenEditState.Disabled);
		nLineBurstVfx.GlobalPosition = position;
		return nLineBurstVfx;
	}

	public override void _Ready()
	{
		base.Emitting = true;
		TaskHelper.RunSafely(DeleteAfterComplete());
	}

	private async Task DeleteAfterComplete()
	{
		await ToSignal(this, GpuParticles2D.SignalName.Finished);
		this.QueueFreeSafely();
	}
}
