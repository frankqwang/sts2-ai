using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;
using MegaCrit.Sts2.Core.Unlocks;

namespace MegaCrit.Sts2.Core.Nodes.Screens.PotionLab;

public partial class NPotionLab : NSubmenu
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/potion_lab/potion_lab");

	private NScrollableContainer _screenContents;

	private NPotionLabCategory _common;

	private NPotionLabCategory _uncommon;

	private NPotionLabCategory _rare;

	private NPotionLabCategory _special;

	private Tween? _screenTween;

	private Task? _loadTask;

	public static string[] AssetPaths => new string[3]
	{
		_scenePath,
		NLabPotionHolder.scenePath,
		NLabPotionHolder.lockedIconPath
	};

	protected override Control? InitialFocusedControl => _common.DefaultFocusedControl;

	public static NPotionLab? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NPotionLab>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		ConnectSignals();
		_screenContents = GetNode<NScrollableContainer>("%ScreenContents");
		_common = GetNode<NPotionLabCategory>("%Common");
		_uncommon = GetNode<NPotionLabCategory>("%Uncommon");
		_rare = GetNode<NPotionLabCategory>("%Rare");
		_special = GetNode<NPotionLabCategory>("%Special");
	}

	public override void OnSubmenuOpened()
	{
		base.OnSubmenuOpened();
		_loadTask = TaskHelper.RunSafely(LoadPotions());
	}

	public override void OnSubmenuClosed()
	{
		base.OnSubmenuClosed();
		_screenTween?.Kill();
		ClearPotions();
	}

	protected override void OnSubmenuShown()
	{
		base.OnSubmenuShown();
		TaskHelper.RunSafely(TweenAfterLoading());
	}

	private async Task TweenAfterLoading()
	{
		_screenContents.Modulate = new Color(1f, 1f, 1f, 0f);
		if (_loadTask != null)
		{
			await _loadTask;
		}
		_screenTween?.Kill();
		_screenTween = CreateTween();
		_screenTween.TweenProperty(_screenContents, "modulate:a", 1f, 0.25).From(0f);
	}

	private async Task LoadPotions()
	{
		_common.Modulate = Colors.Transparent;
		_uncommon.Modulate = Colors.Transparent;
		_rare.Modulate = Colors.Transparent;
		_special.Modulate = Colors.Transparent;
		UnlockState unlockState = SaveManager.Instance.GenerateUnlockStateFromProgress();
		HashSet<PotionModel> allUnlockedPotions = unlockState.Potions.ToHashSet();
		HashSet<PotionModel> seenPotions = SaveManager.Instance.Progress.DiscoveredPotions.Select(ModelDb.GetByIdOrNull<PotionModel>).OfType<PotionModel>().ToHashSet();
		_common.LoadPotions(PotionRarity.Common, new LocString("potion_lab", "COMMON"), seenPotions, unlockState, allUnlockedPotions);
		_uncommon.LoadPotions(PotionRarity.Uncommon, new LocString("potion_lab", "UNCOMMON"), seenPotions, unlockState, allUnlockedPotions);
		_rare.LoadPotions(PotionRarity.Rare, new LocString("potion_lab", "RARE"), seenPotions, unlockState, allUnlockedPotions);
		_special.LoadPotions(PotionRarity.Event, new LocString("potion_lab", "SPECIAL"), seenPotions, unlockState, allUnlockedPotions, PotionRarity.Token);
		List<IReadOnlyList<Control>> list = new List<IReadOnlyList<Control>>();
		list.AddRange(_common.GetGridItems());
		list.AddRange(_uncommon.GetGridItems());
		list.AddRange(_rare.GetGridItems());
		list.AddRange(_special.GetGridItems());
		for (int i = 0; i < list.Count; i++)
		{
			for (int j = 0; j < list[i].Count; j++)
			{
				Control control = list[i][j];
				NodePath path;
				if (j <= 0)
				{
					IReadOnlyList<Control> readOnlyList = list[i];
					path = readOnlyList[readOnlyList.Count - 1].GetPath();
				}
				else
				{
					path = list[i][j - 1].GetPath();
				}
				control.FocusNeighborLeft = path;
				control.FocusNeighborRight = ((j < list[i].Count - 1) ? list[i][j + 1].GetPath() : list[i][0].GetPath());
				if (i > 0)
				{
					control.FocusNeighborTop = ((j < list[i - 1].Count) ? list[i - 1][j].GetPath() : list[i - 1][list[i - 1].Count - 1].GetPath());
				}
				else
				{
					control.FocusNeighborTop = list[i][j].GetPath();
				}
				if (i < list.Count - 1)
				{
					control.FocusNeighborBottom = ((j < list[i + 1].Count) ? list[i + 1][j].GetPath() : list[i + 1][list[i + 1].Count - 1].GetPath());
				}
				else
				{
					control.FocusNeighborBottom = list[i][j].GetPath();
				}
			}
		}
		await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		_common.Modulate = Colors.White;
		_uncommon.Modulate = Colors.White;
		_rare.Modulate = Colors.White;
		_special.Modulate = Colors.White;
		_screenContents.InstantlyScrollToTop();
		InitialFocusedControl?.TryGrabFocus();
	}

	private void ClearPotions()
	{
		_common.ClearPotions();
		_uncommon.ClearPotions();
		_rare.ClearPotions();
		_special.ClearPotions();
	}
}
