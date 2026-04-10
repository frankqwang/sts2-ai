using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Timeline;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Timeline.UnlockScreens;

public partial class NUnlockTimelineScreen : NUnlockScreen
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("timeline_screen/unlock_timeline_screen");

	private List<EpochSlotData> _erasToUnlock;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public static NUnlockTimelineScreen Create()
	{
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NUnlockTimelineScreen>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
	}

	public void SetUnlocks(List<EpochSlotData> eras)
	{
		_erasToUnlock = eras.OrderBy((EpochSlotData a) => a.EraPosition).ToList();
	}

	public override void Open()
	{
		base.Open();
		TaskHelper.RunSafely(AnimateExpansion());
	}

	private async Task AnimateExpansion()
	{
		await NTimelineScreen.Instance.HideBackstopAndShowUi(showBackButton: false);
		await NTimelineScreen.Instance.AddEpochSlots(_erasToUnlock, isAnimated: true);
		NTimelineScreen.Instance.ShowHeaderAndActionsUi();
		NTimelineScreen.Instance.SetScreenDraggability();
		await Close();
		NTimelineScreen.Instance.EnableInput();
	}
}
