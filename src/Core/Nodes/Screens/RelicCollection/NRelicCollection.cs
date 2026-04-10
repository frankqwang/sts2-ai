using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Relics;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;
using MegaCrit.Sts2.Core.Unlocks;

namespace MegaCrit.Sts2.Core.Nodes.Screens.RelicCollection;

public partial class NRelicCollection : NSubmenu
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/relic_collection/relic_collection");

	private NScrollableContainer _screenContents;

	private NRelicCollectionCategory _starter;

	private NRelicCollectionCategory _common;

	private NRelicCollectionCategory _uncommon;

	private NRelicCollectionCategory _rare;

	private NRelicCollectionCategory _shop;

	private NRelicCollectionCategory _ancient;

	private NRelicCollectionCategory _event;

	private Tween? _screenTween;

	private Task? _loadTask;

	private readonly List<RelicModel> _relics = new List<RelicModel>();

	public IReadOnlyList<RelicModel> Relics => _relics;

	public static string[] AssetPaths => new string[4]
	{
		_scenePath,
		NRelicCollectionEntry.scenePath,
		NRelicCollectionCategory.scenePath,
		NRelicCollectionEntry.lockedIconPath
	};

	protected override Control? InitialFocusedControl => _starter.DefaultFocusedControl;

	public static NRelicCollection? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NRelicCollection>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		ConnectSignals();
		_screenContents = GetNode<NScrollableContainer>("%ScreenContents");
		_starter = GetNode<NRelicCollectionCategory>("%Starter");
		_common = GetNode<NRelicCollectionCategory>("%Common");
		_uncommon = GetNode<NRelicCollectionCategory>("%Uncommon");
		_rare = GetNode<NRelicCollectionCategory>("%Rare");
		_shop = GetNode<NRelicCollectionCategory>("%Shop");
		_ancient = GetNode<NRelicCollectionCategory>("%Ancient");
		_event = GetNode<NRelicCollectionCategory>("%Event");
	}

	public override void OnSubmenuOpened()
	{
		base.OnSubmenuOpened();
		_relics.Clear();
		_loadTask = TaskHelper.RunSafely(LoadRelics());
	}

	public override void OnSubmenuClosed()
	{
		base.OnSubmenuClosed();
		_screenTween?.Kill();
		ClearRelics();
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
		_screenTween.TweenProperty(_screenContents, "modulate:a", 1f, 0.4).From(0f);
	}

	private async Task LoadRelics()
	{
		_starter.Modulate = Colors.Transparent;
		_common.Modulate = Colors.Transparent;
		_uncommon.Modulate = Colors.Transparent;
		_rare.Modulate = Colors.Transparent;
		_shop.Modulate = Colors.Transparent;
		_ancient.Modulate = Colors.Transparent;
		_event.Modulate = Colors.Transparent;
		HashSet<RelicModel> seenRelics = SaveManager.Instance.Progress.DiscoveredRelics.Select(ModelDb.GetByIdOrNull<RelicModel>).OfType<RelicModel>().ToHashSet();
		UnlockState unlockState = SaveManager.Instance.GenerateUnlockStateFromProgress();
		HashSet<RelicModel> allUnlockedRelics = unlockState.Relics.ToHashSet();
		_starter.LoadRelics(RelicRarity.Starter, this, new LocString("relic_collection", "STARTER"), seenRelics, unlockState, allUnlockedRelics);
		_common.LoadRelics(RelicRarity.Common, this, new LocString("relic_collection", "COMMON"), seenRelics, unlockState, allUnlockedRelics);
		_uncommon.LoadRelics(RelicRarity.Uncommon, this, new LocString("relic_collection", "UNCOMMON"), seenRelics, unlockState, allUnlockedRelics);
		_rare.LoadRelics(RelicRarity.Rare, this, new LocString("relic_collection", "RARE"), seenRelics, unlockState, allUnlockedRelics);
		_shop.LoadRelics(RelicRarity.Shop, this, new LocString("relic_collection", "SHOP"), seenRelics, unlockState, allUnlockedRelics);
		_ancient.LoadRelics(RelicRarity.Ancient, this, new LocString("relic_collection", "ANCIENT"), seenRelics, unlockState, allUnlockedRelics);
		_event.LoadRelics(RelicRarity.Event, this, new LocString("relic_collection", "EVENT"), seenRelics, unlockState, allUnlockedRelics);
		List<IReadOnlyList<Control>> list = new List<IReadOnlyList<Control>>();
		list.AddRange(_starter.GetGridItems());
		list.AddRange(_common.GetGridItems());
		list.AddRange(_uncommon.GetGridItems());
		list.AddRange(_rare.GetGridItems());
		list.AddRange(_shop.GetGridItems());
		list.AddRange(_ancient.GetGridItems());
		list.AddRange(_event.GetGridItems());
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
		_starter.Modulate = Colors.White;
		_common.Modulate = Colors.White;
		_uncommon.Modulate = Colors.White;
		_rare.Modulate = Colors.White;
		_shop.Modulate = Colors.White;
		_ancient.Modulate = Colors.White;
		_event.Modulate = Colors.White;
		_screenContents.InstantlyScrollToTop();
		InitialFocusedControl?.TryGrabFocus();
	}

	public void AddRelics(IEnumerable<RelicModel> relics)
	{
		_relics.AddRange(relics);
	}

	private void ClearRelics()
	{
		_starter.ClearRelics();
		_common.ClearRelics();
		_uncommon.ClearRelics();
		_rare.ClearRelics();
		_shop.ClearRelics();
		_ancient.ClearRelics();
		_event.ClearRelics();
		_relics.Clear();
	}

	public void SetLastFocusedRelic(NRelicCollectionEntry relic)
	{
		_lastFocusedControl = relic;
	}
}
