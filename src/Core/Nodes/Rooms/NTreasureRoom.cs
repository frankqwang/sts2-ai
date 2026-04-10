using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Ftue;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Rooms;

public partial class NTreasureRoom : Control, IScreenContext, IRoomWithProceedButton
{
	private TreasureRoom _room;

	private IRunState _runState;

	private NCommonBanner _banner;

	private NButton _chestButton;

	private Node2D _chestNode;

	private MegaSprite _chestAnimController;

	private NProceedButton _proceedButton;

	private MegaSkin _regularChestSkin;

	private MegaSkin _outlineChestSkin;

	private GpuParticles2D _goldParticles;

	private NTreasureRoomRelicCollection _relicCollection;

	private static readonly string _scenePath = SceneHelper.GetScenePath("rooms/treasure_room");

	private bool _isRelicCollectionOpen;

	private bool _hasRelicBeenClaimed;

	public NProceedButton ProceedButton => _proceedButton;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public Control? DefaultFocusedControl
	{
		get
		{
			if (!_isRelicCollectionOpen)
			{
				return null;
			}
			return _relicCollection.DefaultFocusedControl;
		}
	}

	public static NTreasureRoom? Create(TreasureRoom room, IRunState runState)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NTreasureRoom nTreasureRoom = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NTreasureRoom>(PackedScene.GenEditState.Disabled);
		nTreasureRoom._room = room;
		nTreasureRoom._runState = runState;
		return nTreasureRoom;
	}

	public override void _Ready()
	{
		_banner = GetNode<NCommonBanner>("%Banner");
		if (_runState.Players.Count == 1)
		{
			_banner.label.SetTextAutoSize(new LocString("gameplay_ui", "TREASURE_BANNER").GetFormattedText());
		}
		else
		{
			_banner.label.SetTextAutoSize(new LocString("gameplay_ui", "CHOOSE_SHARED_RELIC_HEADER").GetFormattedText());
		}
		_proceedButton = GetNode<NProceedButton>("%ProceedButton");
		_chestNode = GetNode<Node2D>("%ChestVisual");
		_chestAnimController = new MegaSprite(_chestNode);
		_goldParticles = GetNode<GpuParticles2D>("%GoldExplosion");
		_relicCollection = GetNode<NTreasureRoomRelicCollection>("%RelicCollection");
		_relicCollection.Initialize(_runState);
		_relicCollection.Visible = false;
		_chestAnimController.SetSkeletonDataRes(_runState.Act.ChestSpineResource);
		MegaSkeleton skeleton = _chestAnimController.GetSkeleton();
		MegaSkeletonDataResource data = skeleton.GetData();
		_regularChestSkin = data.FindSkin(_runState.Act.ChestSpineSkinNameNormal);
		_outlineChestSkin = data.FindSkin(_runState.Act.ChestSpineSkinNameStroke);
		skeleton.SetSlotsToSetupPose();
		_chestAnimController.GetAnimationState().Apply(skeleton);
		MegaAnimationState animationState = _chestAnimController.GetAnimationState();
		animationState.SetAnimation("animation", loop: false);
		_chestAnimController.GetAnimationState().AddAnimation("shine_fade", 0f, loop: false);
		_proceedButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnProceedButtonPressed));
		_proceedButton.UpdateText(NProceedButton.ProceedLoc);
		animationState.SetTimeScale(0f);
		UpdateChestSkin(showOutline: false);
		_chestButton = GetNode<NButton>("%Chest");
		_chestButton.Connect(Control.SignalName.MouseEntered, Callable.From(OnMouseEntered));
		_chestButton.Connect(Control.SignalName.MouseExited, Callable.From(OnMouseExited));
		_chestButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnChestButtonReleased));
	}

	public override void _EnterTree()
	{
		ActiveScreenContext.Instance.Updated += OnActiveScreenChanged;
	}

	public override void _ExitTree()
	{
		ActiveScreenContext.Instance.Updated -= OnActiveScreenChanged;
	}

	private void OnProceedButtonPressed(NButton _)
	{
		TaskHelper.RunSafely(RunManager.Instance.ProceedFromTerminalRewardsScreen());
	}

	private void OnProceedButtonReleased(NButton _)
	{
		NMapScreen.Instance.Open();
	}

	private void OnChestButtonReleased(NButton _)
	{
		TaskHelper.RunSafely(OpenChest());
		_chestButton.Disable();
	}

	private void OnMouseEntered()
	{
		UpdateChestSkin(showOutline: true);
	}

	private void OnMouseExited()
	{
		UpdateChestSkin(showOutline: false);
	}

	private async Task OpenChest()
	{
		_banner.AnimateIn();
		_proceedButton.Disable();
		UpdateChestSkin(showOutline: false);
		SfxCmd.Play(_runState.Act.ChestOpenSfx);
		_chestAnimController.GetAnimationState().SetTimeScale(1f);
		_chestButton.MouseFilter = MouseFilterEnum.Ignore;
		int num = await _room.DoNormalRewards();
		if (num > 0)
		{
			_goldParticles.Amount = num;
			_goldParticles.Emitting = true;
		}
		await _room.DoExtraRewardsIfNeeded();
		_relicCollection.InitializeRelics();
		_relicCollection.AnimIn(_chestNode);
		_isRelicCollectionOpen = true;
		ActiveScreenContext.Instance.Update();
		TaskHelper.RunSafely(RelicFtueCheck());
		await _relicCollection.RelicPickingFinished();
		_isRelicCollectionOpen = false;
		_banner.AnimateOut();
		NMapScreen.Instance.SetTravelEnabled(enabled: true);
		_proceedButton.Enable();
		_hasRelicBeenClaimed = true;
		_relicCollection.AnimOut(_chestNode);
	}

	private async Task RelicFtueCheck()
	{
		if (!SaveManager.Instance.SeenFtue("obtain_relic_ftue"))
		{
			_relicCollection.SetSelectionEnabled(isEnabled: false);
			await Cmd.Wait(1f);
			Control relicReward = ((!_relicCollection.SingleplayerRelicHolder.Visible) ? ((Control)_relicCollection) : ((Control)_relicCollection.SingleplayerRelicHolder));
			_relicCollection.SetSelectionEnabled(isEnabled: true);
			NModalContainer.Instance.Add(NRelicRewardFtue.Create(relicReward));
			SaveManager.Instance.MarkFtueAsComplete("obtain_relic_ftue");
		}
	}

	private void UpdateChestSkin(bool showOutline)
	{
		MegaSkeleton skeleton = _chestAnimController.GetSkeleton();
		skeleton.SetSkin(showOutline ? _outlineChestSkin : _regularChestSkin);
		skeleton.SetSlotsToSetupPose();
		_chestAnimController.GetAnimationState().Apply(skeleton);
	}

	private void OnActiveScreenChanged()
	{
		this.UpdateControllerNavEnabled();
		if (ActiveScreenContext.Instance.IsCurrent(this) && _hasRelicBeenClaimed)
		{
			_proceedButton.Enable();
		}
		else
		{
			_proceedButton.Disable();
		}
	}
}
