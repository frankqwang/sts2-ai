using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NEpochHighlightVfx : Control
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/epoch_highlight_vfx");

	public static NEpochHighlightVfx Create()
	{
		return PreloadManager.Cache.GetScene(scenePath).Instantiate<NEpochHighlightVfx>(PackedScene.GenEditState.Disabled);
	}
}
