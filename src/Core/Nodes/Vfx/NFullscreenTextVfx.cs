using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NFullscreenTextVfx : Control
{
	public static NFullscreenTextVfx? Create(string text)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		Log.Info(text);
		NFullscreenTextVfx nFullscreenTextVfx = PreloadManager.Cache.GetScene("res://scenes/vfx/vfx_fullscreen_text.tscn").Instantiate<NFullscreenTextVfx>(PackedScene.GenEditState.Disabled);
		nFullscreenTextVfx.GetNode<MegaLabel>("Label").SetTextAutoSize(text);
		TaskHelper.RunSafely(nFullscreenTextVfx.SelfDestruct());
		return nFullscreenTextVfx;
	}

	private async Task SelfDestruct()
	{
		await Task.Delay(500);
		this.QueueFreeSafely();
	}
}
