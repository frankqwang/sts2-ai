using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NRainVfx : GpuParticles2D
{
	private static string ScenePath => SceneHelper.GetScenePath("vfx/whole_screen/vfx_rain");

	public static NRainVfx Create()
	{
		return PreloadManager.Cache.GetScene(ScenePath).Instantiate<NRainVfx>(PackedScene.GenEditState.Disabled);
	}
}
