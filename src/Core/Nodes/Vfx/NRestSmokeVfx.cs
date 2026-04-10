using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NRestSmokeVfx : Node2D
{
	private GpuParticles2D _clouds;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	private static string ScenePath => SceneHelper.GetScenePath("vfx/rest_smoke_vfx");

	public static NRestSmokeVfx? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(ScenePath).Instantiate<NRestSmokeVfx>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		_clouds = GetNode<GpuParticles2D>("Clouds");
		_clouds.Emitting = true;
		base.GlobalPosition = GetViewport().GetVisibleRect().GetCenter();
		TaskHelper.RunSafely(DeleteWhenFinished());
	}

	private async Task DeleteWhenFinished()
	{
		await Task.Delay(4000);
		if (GodotObject.IsInstanceValid(this))
		{
			this.QueueFreeSafely();
		}
	}
}
