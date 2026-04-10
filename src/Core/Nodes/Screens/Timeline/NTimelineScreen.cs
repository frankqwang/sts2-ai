using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Achievements;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Nodes.Screens.Timeline.UnlockScreens;
using MegaCrit.Sts2.Core.Nodes.Vfx;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;
using MegaCrit.Sts2.Core.Timeline;
using MegaCrit.Sts2.Core.Timeline.Epochs;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Timeline;

public partial class NTimelineScreen : NSubmenu
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("timeline_screen/timeline_screen");

	private const string _placeEpochSparksPath = "res://scenes/timeline_screen/place_epoch_sparks.tscn";

	private NEpochInspectScreen _inspectScreen;

	private NEpochReminderText _reminderText;

	private Control _reminderVfxHolder;

	private ColorRect _backstop;

	private Control _inputBlocker;

	private Control _lineContainer;

	private Control _line;

	private HBoxContainer _epochSlotContainer;

	private NSlotsContainer _slotsContainer;

	private NBackButton _backButton;

	private ProgressState _save;

	private bool _isUiVisible;

	private Dictionary<EpochEra, NEraColumn> _uniqueEpochEras = new Dictionary<EpochEra, NEraColumn>();

	private NEpochSlot? _queuedInspectScreen;

	private Queue<NUnlockScreen> _unlockScreens = new Queue<NUnlockScreen>();

	private Tween? _lineGrowTween;

	private Tween? _backstopTween;

	public static string[] AssetPaths
	{
		get
		{
			List<string> list = new List<string>();
			list.Add(_scenePath);
			list.Add("res://scenes/timeline_screen/place_epoch_sparks.tscn");
			list.Add(NEpochHighlightVfx.scenePath);
			list.Add(NEpochOffscreenVfx.scenePath);
			list.Add(NEpochInspectScreen.lockedImagePath);
			list.AddRange(NUnlockTimelineScreen.AssetPaths);
			list.AddRange(NUnlockPotionsScreen.AssetPaths);
			list.AddRange(NUnlockRelicsScreen.AssetPaths);
			list.AddRange(NUnlockCardsScreen.AssetPaths);
			list.AddRange(NUnlockMiscScreen.AssetPaths);
			list.AddRange(NUnlockCharacterScreen.AssetPaths);
			list.AddRange(NEraColumn.assetPaths);
			list.AddRange(GetAllEraTexturePaths());
			return list.ToArray();
		}
	}

	public static NTimelineScreen Instance => NGame.Instance.MainMenu.SubmenuStack.GetSubmenuType<NTimelineScreen>();

	protected override Control? InitialFocusedControl => _epochSlotContainer.GetChildren().SelectMany((Node c) => c.GetChildren().OfType<NEpochSlot>()).FirstOrDefault((NEpochSlot s) => s.model is NeowEpoch);

	public static NTimelineScreen? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NTimelineScreen>(PackedScene.GenEditState.Disabled);
	}

	public override void OnSubmenuOpened()
	{
		ResetScreen();
		DisableInput();
		SerializableEpoch serializableEpoch = SaveManager.Instance.Progress.Epochs.FirstOrDefault((SerializableEpoch e) => e.Id == EpochModel.GetId<NeowEpoch>());
		bool flag = serializableEpoch == null;
		bool flag2 = flag;
		if (!flag2)
		{
			EpochState state = serializableEpoch.State;
			bool flag3 = (uint)(state - 1) <= 1u;
			flag2 = flag3;
		}
		if (flag2)
		{
			SaveManager.Instance.Progress.ObtainEpoch(EpochModel.GetId<NeowEpoch>());
		}
		if (SaveManager.Instance.IsNeowDiscovered())
		{
			TaskHelper.RunSafely(FirstTimeLogic());
		}
		else
		{
			SfxCmd.Play("event:/sfx/ui/timeline/ui_timeline_open");
			TaskHelper.RunSafely(InitScreen());
		}
		SetScreenDraggability();
		AchievementsHelper.CheckTimelineComplete();
	}

	public override void OnSubmenuClosed()
	{
		base.OnSubmenuClosed();
		ResetScreen();
	}

	protected override void OnSubmenuShown()
	{
		base.ProcessMode = ProcessModeEnum.Inherit;
		RefreshBackButton();
	}

	protected override void OnSubmenuHidden()
	{
		base.ProcessMode = ProcessModeEnum.Disabled;
		NGame.Instance.MainMenu?.RefreshButtons();
	}

	public override void _Ready()
	{
		ConnectSignals();
		_epochSlotContainer = GetNode<HBoxContainer>("%EpochSlots");
		_reminderText = GetNode<NEpochReminderText>("%EpochReminderText");
		_reminderVfxHolder = GetNode<Control>("%ReminderVfxHolder");
		_inputBlocker = GetNode<Control>("%InputBlocker");
		_backstop = GetNode<ColorRect>("%SharedBackstop");
		_inspectScreen = GetNode<NEpochInspectScreen>("%EpochInspectScreen");
		_line = GetNode<Control>("%Line");
		_lineContainer = GetNode<Control>("%LineContainer");
		_slotsContainer = GetNode<NSlotsContainer>("%SlotsContainer");
		_backButton = GetNode<NBackButton>("BackButton");
		_backButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnBackButtonPressed));
		_save = SaveManager.Instance.Progress;
		Tween tween = CreateTween();
		tween.TweenProperty(_slotsContainer, "modulate:a", 1f, 1.0);
	}

	private static void OnBackButtonPressed(NButton obj)
	{
		SfxCmd.Play("event:/sfx/ui/map/map_close");
	}

	private async Task FirstTimeLogic()
	{
		await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		NTimelineTutorial nTimelineTutorial = SceneHelper.Instantiate<NTimelineTutorial>("timeline_screen/timeline_tutorial");
		this.AddChildSafely(nTimelineTutorial);
		nTimelineTutorial.Init(this);
	}

	public async Task SpawnFirstTimeTimeline()
	{
		SfxCmd.Play("event:/sfx/ui/timeline/ui_timeline_open");
		Log.Info("Running first time logic");
		List<EpochSlotData> slotsToAdd = new List<EpochSlotData>(1)
		{
			new EpochSlotData(EpochModel.GetId<NeowEpoch>(), EpochSlotState.Obtained)
		};
		await AddEpochSlots(slotsToAdd, isAnimated: true);
		SaveManager.Instance.UnlockSlot(EpochModel.GetId<NeowEpoch>());
		EnableInput();
	}

	private async Task InitScreen()
	{
		Log.Info("Initializing Timeline:");
		List<EpochSlotData> list = new List<EpochSlotData>();
		_lineGrowTween?.Kill();
		_lineGrowTween = CreateTween();
		_lineGrowTween.TweenProperty(_lineContainer, "modulate:a", 1f, 0.5);
		foreach (SerializableEpoch epoch in _save.Epochs)
		{
			if (epoch.State != EpochState.ObtainedNoSlot)
			{
				list.Add(new EpochSlotData(epoch.Id, EpochSlotState.NotObtained));
			}
		}
		list = list.OrderBy((EpochSlotData a) => a.EraPosition).ToList();
		await AddEpochSlots(list, isAnimated: false);
		int num = 0;
		foreach (SerializableEpoch epoch2 in _save.Epochs)
		{
			EpochModel epochModel = EpochModel.Get(epoch2.Id);
			if (epoch2.State <= EpochState.ObtainedNoSlot)
			{
				continue;
			}
			foreach (Node child in _uniqueEpochEras[epochModel.Era].GetChildren())
			{
				if (child is NEpochSlot nEpochSlot && nEpochSlot.eraPosition == epochModel.EraPosition)
				{
					num++;
					nEpochSlot.SetState((epoch2.State >= EpochState.Revealed) ? EpochSlotState.Complete : EpochSlotState.Obtained);
					TaskHelper.RunSafely(child.GetParent<NEraColumn>().SpawnNameAndYear());
				}
			}
		}
		Log.Info($"{num} Epochs are complete");
		TaskHelper.RunSafely(NavigateToRevealableSlot());
	}

	private async Task NavigateToRevealableSlot()
	{
		if (SaveManager.Instance.GetDiscoveredEpochCount() == 0)
		{
			EnableInput();
			return;
		}
		await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		float getInitX = _slotsContainer.GetInitX;
		float slotPositionX = 0f;
		float num = float.MaxValue;
		foreach (NEraColumn value in _uniqueEpochEras.Values)
		{
			foreach (NEpochSlot item in value.GetChildrenRecursive<NEpochSlot>())
			{
				if (item.State == EpochSlotState.Obtained)
				{
					float num2 = Math.Abs(getInitX - item.GlobalPosition.X);
					if (num2 < num)
					{
						slotPositionX = item.GlobalPosition.X;
						num = num2;
					}
				}
			}
		}
		await TaskHelper.RunSafely(_slotsContainer.LerpToSlot(slotPositionX));
		EnableInput();
	}

	public async Task AddEpochSlots(List<EpochSlotData> slotsToAdd, bool isAnimated)
	{
		List<NEraColumn> list = new List<NEraColumn>();
		if (isAnimated)
		{
			foreach (NEraColumn value2 in _uniqueEpochEras.Values)
			{
				TaskHelper.RunSafely(value2.SaveBeforeAnimationPosition());
			}
		}
		foreach (EpochSlotData item in slotsToAdd)
		{
			if (!_uniqueEpochEras.TryGetValue(item.Era, out NEraColumn value))
			{
				NEraColumn nEraColumn = NEraColumn.Create(item);
				_epochSlotContainer.AddChildSafely(nEraColumn);
				int toIndex = 0;
				foreach (Node child in _epochSlotContainer.GetChildren())
				{
					if (child is NEraColumn nEraColumn2 && nEraColumn.era > nEraColumn2.era)
					{
						toIndex = nEraColumn2.GetIndex() + 1;
					}
				}
				_epochSlotContainer.MoveChild(nEraColumn, toIndex);
				list.Add(nEraColumn);
				_uniqueEpochEras.Add(item.Era, nEraColumn);
			}
			else
			{
				value.AddSlot(item);
			}
		}
		Log.Info($" Created {slotsToAdd.Count} Epoch slots");
		Log.Info($" Created {list.Count} Era columns");
		if (isAnimated)
		{
			List<Vector2> list2 = PredictHBoxLayout(_epochSlotContainer);
			foreach (NEraColumn value3 in _uniqueEpochEras.Values)
			{
				value3.SetPredictedPosition(list2[value3.GetIndex()]);
			}
			await GrowTimelineAndAddEraIcons(list);
		}
		else
		{
			InitLineAndIcons(list);
		}
	}

	private List<Vector2> PredictHBoxLayout(HBoxContainer hbox)
	{
		float num = 0f;
		float num2 = hbox.GetThemeConstant(ThemeConstants.BoxContainer.separation, "HBoxContainer");
		List<Control> list = (from c in hbox.GetChildren().OfType<Control>()
			where c.Visible
			select c).ToList();
		int num3 = 0;
		foreach (Control item in list)
		{
			num += item.CustomMinimumSize.X;
			if ((item.SizeFlagsHorizontal & SizeFlags.Expand) != SizeFlags.ShrinkBegin)
			{
				num3++;
			}
		}
		num += num2 * (float)Math.Max(list.Count - 1, 0);
		float num4 = hbox.Size.X - num;
		float num5 = ((num3 > 0) ? (num4 / (float)num3) : 0f);
		float num6 = 0f;
		List<Vector2> list2 = new List<Vector2>();
		foreach (Control item2 in list)
		{
			float num7 = item2.CustomMinimumSize.X;
			if ((item2.SizeFlagsHorizontal & SizeFlags.Expand) != SizeFlags.ShrinkBegin)
			{
				num7 += num5;
			}
			list2.Add(new Vector2(num6, 0f));
			num6 += num7 + num2;
		}
		return list2;
	}

	private async Task GrowTimelineAndAddEraIcons(List<NEraColumn> newlyCreatedColumns)
	{
		if (newlyCreatedColumns.Count > 0)
		{
			_lineGrowTween?.Kill();
			_lineGrowTween = CreateTween().SetParallel();
			_lineGrowTween.TweenProperty(_lineContainer, "modulate:a", 1f, 0.5);
			_lineGrowTween.TweenProperty(_line, "custom_minimum_size:x", (float)_uniqueEpochEras.Count * 226f, 2.0).SetEase(Tween.EaseType.InOut).SetTrans(Tween.TransitionType.Cubic);
			await ToSignal(_lineGrowTween, Tween.SignalName.Finished);
			Log.Info("Spawning slots...");
			foreach (NEraColumn newlyCreatedColumn in newlyCreatedColumns)
			{
				newlyCreatedColumn.SpawnIcon();
			}
			newlyCreatedColumns.Clear();
		}
		foreach (NEraColumn value in _uniqueEpochEras.Values)
		{
			TaskHelper.RunSafely(value.SpawnSlots(isAnimated: true));
		}
	}

	private void InitLineAndIcons(List<NEraColumn> newlyCreatedColumns)
	{
		if (newlyCreatedColumns.Count == 0)
		{
			return;
		}
		_line.CustomMinimumSize = new Vector2((float)_uniqueEpochEras.Count * 226f, _line.CustomMinimumSize.Y);
		foreach (NEraColumn newlyCreatedColumn in newlyCreatedColumns)
		{
			newlyCreatedColumn.SpawnIcon();
		}
		foreach (NEraColumn value in _uniqueEpochEras.Values)
		{
			TaskHelper.RunSafely(value.SpawnSlots(isAnimated: false));
		}
		newlyCreatedColumns.Clear();
	}

	public static (Texture2D Texture, string Name) GetEraIcon(EpochEra era)
	{
		return (Texture: PreloadManager.Cache.GetTexture2D(GetEraTexturePath(era)), Name: StringHelper.Slugify(era.ToString()));
	}

	private static string GetEraTexturePath(EpochEra era)
	{
		if (era >= EpochEra.Seeds0)
		{
			return $"res://images/atlases/era_atlas.sprites/era_{(int)era}.tres";
		}
		return $"res://images/atlases/era_atlas.sprites/era_minus_{Math.Abs((int)era)}.tres";
	}

	private static IEnumerable<string> GetAllEraTexturePaths()
	{
		EpochEra[] values = Enum.GetValues<EpochEra>();
		foreach (EpochEra era in values)
		{
			yield return GetEraTexturePath(era);
		}
	}

	private NEpochSlot? GetSlot(EpochEra era, int position)
	{
		foreach (KeyValuePair<EpochEra, NEraColumn> uniqueEpochEra in _uniqueEpochEras)
		{
			if (uniqueEpochEra.Key == era)
			{
				return (NEpochSlot)uniqueEpochEra.Value.GetChild(uniqueEpochEra.Value.GetChildCount() - position - 2);
			}
		}
		Log.Error($"Could not find Epoch slot: {era}, {position}");
		return null;
	}

	public void OpenInspectScreen(NEpochSlot slot, bool playAnimation)
	{
		if (playAnimation)
		{
			TaskHelper.RunSafely(slot.GetParent<NEraColumn>().SpawnNameAndYear());
		}
		_lastFocusedControl = slot;
		TaskHelper.RunSafely(_inspectScreen.Open(slot, slot.model, playAnimation));
	}

	public void QueueMiscUnlock(string text)
	{
		NUnlockMiscScreen nUnlockMiscScreen = NUnlockMiscScreen.Create();
		nUnlockMiscScreen.SetUnlocks(text);
		_unlockScreens.Enqueue(nUnlockMiscScreen);
	}

	public void QueueCharacterUnlock<T>(EpochModel epoch) where T : CharacterModel
	{
		_unlockScreens.Enqueue(NUnlockCharacterScreen.Create(epoch, ModelDb.Character<T>()));
	}

	public void QueueCardUnlock(IReadOnlyList<CardModel> cards)
	{
		NUnlockCardsScreen nUnlockCardsScreen = NUnlockCardsScreen.Create();
		nUnlockCardsScreen.SetCards(cards);
		_unlockScreens.Enqueue(nUnlockCardsScreen);
	}

	public void QueueRelicUnlock(List<RelicModel> relics)
	{
		NUnlockRelicsScreen nUnlockRelicsScreen = NUnlockRelicsScreen.Create();
		nUnlockRelicsScreen.SetRelics(relics);
		_unlockScreens.Enqueue(nUnlockRelicsScreen);
	}

	public void QueuePotionUnlock(List<PotionModel> potions)
	{
		NUnlockPotionsScreen nUnlockPotionsScreen = NUnlockPotionsScreen.Create();
		nUnlockPotionsScreen.SetPotions(potions);
		_unlockScreens.Enqueue(nUnlockPotionsScreen);
	}

	public void QueueTimelineExpansion(List<EpochSlotData> eraData)
	{
		NUnlockTimelineScreen nUnlockTimelineScreen = NUnlockTimelineScreen.Create();
		nUnlockTimelineScreen.SetUnlocks(eraData);
		_unlockScreens.Enqueue(nUnlockTimelineScreen);
	}

	public void SetScreenDraggability()
	{
		_slotsContainer.MouseFilter = (MouseFilterEnum)((_save.Epochs.Count <= 4) ? 2 : 0);
	}

	public void ShowBackstopAndHideUi()
	{
		_backstop.Visible = true;
		_backstopTween?.Kill();
		_backstopTween = CreateTween().SetParallel();
		_backstopTween.TweenProperty(_slotsContainer, "modulate:a", 0.1f, 0.4);
		_backstopTween.TweenProperty(_backstop, "modulate:a", 0.5f, 0.4);
		_backButton.Disable();
		_reminderText.AnimateOut();
	}

	public async Task HideBackstopAndShowUi(bool showBackButton)
	{
		_backstopTween?.FastForwardToCompletion();
		_backstopTween = CreateTween().SetParallel();
		_backstopTween.TweenProperty(_slotsContainer, "modulate:a", 1f, 0.4).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		_backstopTween.TweenProperty(_backstop, "modulate:a", 0f, 0.4).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		if (showBackButton)
		{
			RefreshBackButton();
		}
		await ToSignal(_backstopTween, Tween.SignalName.Finished);
		_backstop.Visible = false;
	}

	public void OpenQueuedScreen()
	{
		NUnlockScreen nUnlockScreen = _unlockScreens.Dequeue();
		this.AddChildSafely(nUnlockScreen);
		nUnlockScreen.Open();
	}

	public bool IsScreenQueued()
	{
		return _unlockScreens.Count > 0;
	}

	private bool IsInspectScreenQueued()
	{
		return _queuedInspectScreen != null;
	}

	private async Task SpawnEraLabel(EpochEra era)
	{
		await _uniqueEpochEras[era].SpawnNameAndYear();
	}

	public void ShowHeaderAndActionsUi()
	{
		if (!_isUiVisible)
		{
			_isUiVisible = true;
		}
	}

	public void DisableInput()
	{
		_inputBlocker.Visible = true;
		_inputBlocker.MouseFilter = MouseFilterEnum.Stop;
		_slotsContainer.SetEnabled(enabled: false);
		GetViewport().GuiReleaseFocus();
	}

	public void EnableInput()
	{
		if (_queuedInspectScreen == null && _unlockScreens.Count == 0)
		{
			RefreshBackButton();
			_inputBlocker.Visible = false;
			_inputBlocker.MouseFilter = MouseFilterEnum.Ignore;
			_slotsContainer.SetEnabled(enabled: true);
			ActiveScreenContext.Instance.Update();
		}
	}

	private void RefreshBackButton()
	{
		if (SaveManager.Instance.GetDiscoveredEpochCount() > 0)
		{
			if (_epochSlotContainer.GetChildCount() > 1)
			{
				_reminderText.AnimateIn();
			}
			_backButton.Disable();
		}
		else
		{
			_backButton.Enable();
		}
	}

	private void ResetScreen()
	{
		Control lineContainer = _lineContainer;
		Color modulate = _lineContainer.Modulate;
		modulate.A = 0f;
		lineContainer.Modulate = modulate;
		Log.Info("Cleaning up Timeline screen...");
		_uniqueEpochEras = new Dictionary<EpochEra, NEraColumn>();
		_queuedInspectScreen = null;
		_unlockScreens = new Queue<NUnlockScreen>();
		_epochSlotContainer.FreeChildren();
		_slotsContainer.Reset();
	}

	public Control GetReminderVfxHolder()
	{
		return _reminderVfxHolder;
	}
}
