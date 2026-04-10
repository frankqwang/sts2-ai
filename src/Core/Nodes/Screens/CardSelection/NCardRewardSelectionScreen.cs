using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Rewards;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Ftue;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;

public partial class NCardRewardSelectionScreen : Control, IOverlayScreen, IScreenContext
{
	private const ulong _noSelectionTimeMsec = 350uL;

	private Control _ui;

	private NCommonBanner _banner;

	private Control _cardRow;

	private IReadOnlyList<CardCreationResult> _options;

	private IReadOnlyList<CardRewardAlternative> _extraOptions;

	private Control _rewardAlternativesContainer;

	private Control _inspectPrompt;

	private TaskCompletionSource<Tuple<IEnumerable<NCardHolder>, bool>>? _completionSource;

	private Tween? _cardTween;

	private Tween? _buttonTween;

	private const float _cardXOffset = 350f;

	private static readonly Vector2 _bannerAnimPosOffset = new Vector2(0f, 50f);

	private Control? _lastFocusedControl;

	private static string ScenePath => SceneHelper.GetScenePath("screens/card_selection/card_reward_selection_screen");

	public static IEnumerable<string> AssetPaths => new string[1] { ScenePath }.Concat(NCardRewardAlternativeButton.AssetPaths);

	public NetScreenType ScreenType => NetScreenType.CardSelection;

	public bool UseSharedBackstop => true;

	public Control DefaultFocusedControl
	{
		get
		{
			if (_lastFocusedControl != null)
			{
				return _lastFocusedControl;
			}
			List<NGridCardHolder> list = _cardRow.GetChildren().OfType<NGridCardHolder>().ToList();
			return list[list.Count / 2];
		}
	}

	public static NCardRewardSelectionScreen? ShowScreen(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> extraOptions)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCardRewardSelectionScreen nCardRewardSelectionScreen = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NCardRewardSelectionScreen>(PackedScene.GenEditState.Disabled);
		nCardRewardSelectionScreen.Name = "NCardRewardSelectionScreen";
		nCardRewardSelectionScreen._options = options;
		nCardRewardSelectionScreen._extraOptions = extraOptions;
		NOverlayStack.Instance.Push(nCardRewardSelectionScreen);
		return nCardRewardSelectionScreen;
	}

	public override void _Ready()
	{
		_ui = GetNode<Control>("UI");
		_cardRow = GetNode<Control>("UI/CardRow");
		_banner = GetNode<NCommonBanner>("UI/Banner");
		_rewardAlternativesContainer = GetNode<Control>("UI/RewardAlternatives");
		_inspectPrompt = GetNode<Control>("%InspectPrompt");
		_banner.label.SetTextAutoSize(new LocString("gameplay_ui", "CHOOSE_CARD_HEADER").GetRawText());
		_banner.AnimateIn();
		RefreshOptions(_options, _extraOptions);
		UpdateControllerIcons();
		NControllerManager.Instance.Connect(NControllerManager.SignalName.MouseDetected, Callable.From(UpdateControllerIcons));
		NControllerManager.Instance.Connect(NControllerManager.SignalName.ControllerDetected, Callable.From(UpdateControllerIcons));
		NInputManager.Instance.Connect(NInputManager.SignalName.InputRebound, Callable.From(UpdateControllerIcons));
	}

	public void RefreshOptions(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> extraOptions)
	{
		_options = options;
		_extraOptions = extraOptions;
		Vector2 vector = Vector2.Left * (_options.Count - 1) * 350f * 0.5f;
		_lastFocusedControl = null;
		foreach (NGridCardHolder item in _cardRow.GetChildren().OfType<NGridCardHolder>())
		{
			item.QueueFreeSafely();
		}
		foreach (NCardRewardAlternativeButton item2 in _rewardAlternativesContainer.GetChildren().OfType<NCardRewardAlternativeButton>())
		{
			item2.QueueFreeSafely();
		}
		_cardTween = CreateTween().SetParallel();
		for (int i = 0; i < _options.Count; i++)
		{
			CardCreationResult cardCreationResult = _options[i];
			NCard nCard = NCard.Create(cardCreationResult.Card);
			NGridCardHolder holder = NGridCardHolder.Create(nCard);
			_cardRow.AddChildSafely(holder);
			holder.Connect(NCardHolder.SignalName.Pressed, Callable.From<NCardHolder>(SelectCard));
			holder.Connect(NCardHolder.SignalName.AltPressed, Callable.From<NCardHolder>(InspectCard));
			holder.Connect(Control.SignalName.FocusEntered, Callable.From(() => _lastFocusedControl = holder));
			nCard.UpdateVisuals(PileType.None, CardPreviewMode.Normal);
			holder.Scale = holder.SmallScale;
			_cardTween.TweenProperty(holder, "position", vector + Vector2.Right * 350f * i, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			_cardTween.TweenProperty(holder, "modulate", Colors.White, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic)
				.From(Colors.Black);
			nCard.ActivateRewardScreenGlow();
			foreach (RelicModel modifyingRelic in cardCreationResult.ModifyingRelics)
			{
				modifyingRelic.Flash();
				nCard.FlashRelicOnCard(modifyingRelic);
			}
		}
		foreach (CardRewardAlternative rewardOption in _extraOptions)
		{
			NCardRewardAlternativeButton nCardRewardAlternativeButton = NCardRewardAlternativeButton.Create(rewardOption.Title.GetFormattedText(), rewardOption.Hotkey);
			_rewardAlternativesContainer.AddChildSafely(nCardRewardAlternativeButton);
			nCardRewardAlternativeButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(delegate
			{
				OnAlternateRewardSelected(rewardOption.AfterSelected);
				TaskHelper.RunSafely(rewardOption.OnSelect());
			}));
		}
		List<NGridCardHolder> list = _cardRow.GetChildren().OfType<NGridCardHolder>().ToList();
		for (int num = 0; num < _cardRow.GetChildCount(); num++)
		{
			Control child = _cardRow.GetChild<Control>(num);
			child.FocusNeighborBottom = child.GetPath();
			child.FocusNeighborTop = child.GetPath();
			child.FocusNeighborLeft = ((num > 0) ? _cardRow.GetChild(num - 1).GetPath() : _cardRow.GetChild(_cardRow.GetChildCount() - 1).GetPath());
			child.FocusNeighborRight = ((num < _cardRow.GetChildCount() - 1) ? _cardRow.GetChild(num + 1).GetPath() : _cardRow.GetChild(0).GetPath());
		}
		if (NControllerManager.Instance.IsUsingController)
		{
			list[list.Count / 2].TryGrabFocus();
		}
	}

	public override void _ExitTree()
	{
		TaskCompletionSource<Tuple<IEnumerable<NCardHolder>, bool>> completionSource = _completionSource;
		if (completionSource != null)
		{
			Task<Tuple<IEnumerable<NCardHolder>, bool>> task = completionSource.Task;
			if (task != null && !task.IsCompleted)
			{
				_completionSource.SetResult(new Tuple<IEnumerable<NCardHolder>, bool>(Array.Empty<NCardHolder>(), item2: false));
			}
		}
		foreach (NGridCardHolder item in _cardRow.GetChildren().OfType<NGridCardHolder>())
		{
			item.QueueFreeSafely();
		}
	}

	private void OnAlternateRewardSelected(PostAlternateCardRewardAction afterSelected)
	{
		if ((afterSelected != PostAlternateCardRewardAction.None && afterSelected != PostAlternateCardRewardAction.DoNothing) || 1 == 0)
		{
			_completionSource?.SetResult(new Tuple<IEnumerable<NCardHolder>, bool>(Array.Empty<NCardHolder>(), afterSelected == PostAlternateCardRewardAction.DismissScreenAndRemoveReward));
		}
	}

	private void SelectCard(NCardHolder cardHolder)
	{
		if (_completionSource == null)
		{
			throw new InvalidOperationException("CardsSelected must be awaited before a card is selected!");
		}
		_completionSource.SetResult(new Tuple<IEnumerable<NCardHolder>, bool>(new global::_003C_003Ez__ReadOnlySingleElementList<NCardHolder>(cardHolder), item2: true));
	}

	private void InspectCard(NCardHolder cardHolder)
	{
		if (!_completionSource.Task.IsCompleted)
		{
			NInspectCardScreen inspectCardScreen = NGame.Instance.GetInspectCardScreen();
			int num = 1;
			List<CardModel> list = new List<CardModel>(num);
			CollectionsMarshal.SetCount(list, num);
			Span<CardModel> span = CollectionsMarshal.AsSpan(list);
			int index = 0;
			span[index] = cardHolder.CardNode.Model;
			inspectCardScreen.Open(list, 0);
		}
	}

	public async Task<Tuple<IEnumerable<NCardHolder>, bool>> CardsSelected()
	{
		_completionSource = new TaskCompletionSource<Tuple<IEnumerable<NCardHolder>, bool>>();
		return await _completionSource.Task;
	}

	public void AfterOverlayOpened()
	{
		PowerCardFtueCheck();
		_banner.AnimateIn();
		_buttonTween = CreateTween();
		_buttonTween.SetParallel();
		_buttonTween.TweenProperty(_rewardAlternativesContainer, "position", _rewardAlternativesContainer.Position, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back)
			.From(_rewardAlternativesContainer.Position - _bannerAnimPosOffset);
		TaskHelper.RunSafely(DisableCardsForShortTimeAfterOpening());
	}

	private async Task DisableCardsForShortTimeAfterOpening()
	{
		foreach (NGridCardHolder item in _cardRow.GetChildren().OfType<NGridCardHolder>())
		{
			item.SetClickable(isClickable: false);
		}
		await Cmd.Wait(0.35f);
		if (!_cardRow.IsValid())
		{
			return;
		}
		foreach (NGridCardHolder item2 in _cardRow.GetChildren().OfType<NGridCardHolder>())
		{
			item2.SetClickable(isClickable: true);
		}
	}

	private void PowerCardFtueCheck()
	{
		if (!SaveManager.Instance.SeenFtue("power_card_ftue"))
		{
			IEnumerable<NGridCardHolder> source = _cardRow.GetChildren().OfType<NGridCardHolder>();
			NGridCardHolder nGridCardHolder = source.FirstOrDefault((NGridCardHolder h) => h.CardModel.Type == CardType.Power);
			if (nGridCardHolder != null)
			{
				NModalContainer.Instance.Add(NPowerCardFtue.Create(nGridCardHolder));
				SaveManager.Instance.MarkFtueAsComplete("power_card_ftue");
			}
		}
	}

	public void AfterOverlayClosed()
	{
		this.QueueFreeSafely();
	}

	public void AfterOverlayShown()
	{
		base.Visible = true;
	}

	public void AfterOverlayHidden()
	{
		base.Visible = false;
	}

	private void UpdateControllerIcons()
	{
		_inspectPrompt.Visible = NControllerManager.Instance.IsUsingController;
		_inspectPrompt.GetNode<TextureRect>("ControllerIcon").Texture = NInputManager.Instance.GetHotkeyIcon(MegaInput.accept);
		_inspectPrompt.GetNode<MegaLabel>("Label").SetTextAutoSize(new LocString("gameplay_ui", "TO_INSPECT_PROMPT").GetFormattedText());
	}
}
