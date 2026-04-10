using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;

public partial class NChooseACardSelectionScreen : Control, IOverlayScreen, IScreenContext, ICardSelector
{
	private const float _cardXSpacing = 340f;

	private const ulong _noSelectionTimeMsec = 350uL;

	private NCommonBanner _banner;

	private Control _cardRow;

	private NChoiceSelectionSkipButton _skipButton;

	private NCombatPilesContainer _combatPiles;

	private Control _inspectPrompt;

	private NPeekButton _peekButton;

	private readonly TaskCompletionSource<IEnumerable<CardModel>> _completionSource = new TaskCompletionSource<IEnumerable<CardModel>>();

	private ulong _openedTicks;

	private bool _screenComplete;

	private bool _cardSelected;

	private bool _canSkip;

	private Tween? _cardTween;

	private Tween? _fadeTween;

	private IReadOnlyList<CardModel> _cards;

	private static string ScenePath => SceneHelper.GetScenePath("screens/card_selection/choose_a_card_selection_screen");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	public NetScreenType ScreenType => NetScreenType.CardSelection;

	public bool UseSharedBackstop => true;

	public Control DefaultFocusedControl
	{
		get
		{
			if (_peekButton.IsPeeking)
			{
				return NCombatRoom.Instance.DefaultFocusedControl;
			}
			List<NGridCardHolder> list = _cardRow.GetChildren().OfType<NGridCardHolder>().ToList();
			return list[list.Count / 2];
		}
	}

	public static NChooseACardSelectionScreen? ShowScreen(IReadOnlyList<CardModel> cards, bool canSkip)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NChooseACardSelectionScreen nChooseACardSelectionScreen = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NChooseACardSelectionScreen>(PackedScene.GenEditState.Disabled);
		nChooseACardSelectionScreen.Name = "NChooseACardSelectionScreen";
		nChooseACardSelectionScreen._cards = cards;
		nChooseACardSelectionScreen._canSkip = canSkip;
		NOverlayStack.Instance.Push(nChooseACardSelectionScreen);
		return nChooseACardSelectionScreen;
	}

	public override void _Ready()
	{
		_banner = GetNode<NCommonBanner>("Banner");
		_banner.label.SetTextAutoSize(new LocString("gameplay_ui", "CHOOSE_CARD_HEADER").GetRawText());
		_banner.AnimateIn();
		_cardRow = GetNode<Control>("CardRow");
		_combatPiles = GetNode<NCombatPilesContainer>("%CombatPiles");
		if (CombatManager.Instance.IsInProgress)
		{
			_combatPiles.Initialize(_cards.First().Owner);
		}
		_combatPiles.Disable();
		_combatPiles.Visible = false;
		_inspectPrompt = GetNode<Control>("%InspectPrompt");
		Vector2 vector = Vector2.Left * (_cards.Count - 1) * 340f * 0.5f;
		_cardTween = CreateTween().SetParallel();
		for (int i = 0; i < _cards.Count; i++)
		{
			CardModel card = _cards[i];
			NCard nCard = NCard.Create(card);
			NGridCardHolder nGridCardHolder = NGridCardHolder.Create(nCard);
			_cardRow.AddChildSafely(nGridCardHolder);
			nGridCardHolder.Connect(NCardHolder.SignalName.Pressed, Callable.From<NCardHolder>(SelectHolder));
			nGridCardHolder.Connect(NCardHolder.SignalName.AltPressed, Callable.From<NCardHolder>(OpenPreviewScreen));
			nCard.UpdateVisuals(PileType.None, CardPreviewMode.Normal);
			nGridCardHolder.Scale = nGridCardHolder.SmallScale;
			_cardTween.TweenProperty(nGridCardHolder, "position", vector + Vector2.Right * 340f * i, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			_cardTween.TweenProperty(nGridCardHolder, "modulate", Colors.White, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic)
				.From(Colors.Black);
			nCard.ActivateRewardScreenGlow();
		}
		_skipButton = GetNode<NChoiceSelectionSkipButton>("SkipButton");
		if (_canSkip)
		{
			_skipButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnSkipButtonReleased));
			_skipButton.AnimateIn();
		}
		else
		{
			_skipButton.Disable();
			_skipButton.Visible = false;
		}
		_peekButton = GetNode<NPeekButton>("%PeekButton");
		_peekButton.AddTargets(_banner, _cardRow, _skipButton, _inspectPrompt);
		_peekButton.Connect(NPeekButton.SignalName.Toggled, Callable.From<NPeekButton>(delegate
		{
			if (_peekButton.IsPeeking)
			{
				base.MouseFilter = MouseFilterEnum.Ignore;
				_combatPiles.Visible = true;
				_combatPiles.Enable();
				_skipButton.Disable();
			}
			else
			{
				base.MouseFilter = MouseFilterEnum.Stop;
				_combatPiles.Visible = false;
				_combatPiles.Disable();
				ActiveScreenContext.Instance.Update();
				if (_canSkip)
				{
					_skipButton.Enable();
				}
			}
		}));
		for (int num = 0; num < _cardRow.GetChildCount(); num++)
		{
			Control child = _cardRow.GetChild<Control>(num);
			child.FocusNeighborBottom = child.GetPath();
			child.FocusNeighborTop = child.GetPath();
			child.FocusNeighborLeft = ((num > 0) ? _cardRow.GetChild(num - 1).GetPath() : _cardRow.GetChild(_cardRow.GetChildCount() - 1).GetPath());
			child.FocusNeighborRight = ((num < _cardRow.GetChildCount() - 1) ? _cardRow.GetChild(num + 1).GetPath() : _cardRow.GetChild(0).GetPath());
		}
		UpdateControllerIcons();
		NControllerManager.Instance.Connect(NControllerManager.SignalName.MouseDetected, Callable.From(UpdateControllerIcons));
		NControllerManager.Instance.Connect(NControllerManager.SignalName.ControllerDetected, Callable.From(UpdateControllerIcons));
		NInputManager.Instance.Connect(NInputManager.SignalName.InputRebound, Callable.From(UpdateControllerIcons));
	}

	public override void _ExitTree()
	{
		if (!_completionSource.Task.IsCompleted)
		{
			_completionSource.SetCanceled();
		}
		foreach (NGridCardHolder item in _cardRow.GetChildren().OfType<NGridCardHolder>())
		{
			item.QueueFreeSafely();
		}
	}

	private void SelectHolder(NCardHolder cardHolder)
	{
		if (_completionSource == null)
		{
			throw new InvalidOperationException("CardsSelected must be awaited before a card is selected!");
		}
		if (Time.GetTicksMsec() - _openedTicks > 350)
		{
			CardModel cardModel = cardHolder.CardModel;
			_screenComplete = true;
			_cardSelected = true;
			_completionSource.SetResult(new CardModel[1] { cardModel });
		}
	}

	private void OpenPreviewScreen(NCardHolder cardHolder)
	{
		NInspectCardScreen inspectCardScreen = NGame.Instance.GetInspectCardScreen();
		int num = 1;
		List<CardModel> list = new List<CardModel>(num);
		CollectionsMarshal.SetCount(list, num);
		Span<CardModel> span = CollectionsMarshal.AsSpan(list);
		int index = 0;
		span[index] = cardHolder.CardModel;
		inspectCardScreen.Open(list, 0);
	}

	public async Task<IEnumerable<CardModel>> CardsSelected()
	{
		IEnumerable<CardModel> result = await _completionSource.Task;
		NOverlayStack.Instance.Remove(this);
		return result;
	}

	private void OnSkipButtonReleased(NButton _)
	{
		_screenComplete = true;
		_completionSource.SetResult(Array.Empty<CardModel>());
	}

	public void AfterOverlayOpened()
	{
		base.Modulate = Colors.Transparent;
		_openedTicks = Time.GetTicksMsec();
		_fadeTween?.Kill();
		_fadeTween = CreateTween();
		_fadeTween.TweenProperty(this, "modulate:a", 1f, 0.2);
	}

	public void AfterOverlayClosed()
	{
		_fadeTween?.Kill();
		_peekButton.SetPeeking(isPeeking: false);
		this.QueueFreeSafely();
	}

	public void AfterOverlayShown()
	{
		base.Visible = true;
		_peekButton.Enable();
		if (_canSkip && !_peekButton.IsPeeking)
		{
			_skipButton.Enable();
		}
	}

	public void AfterOverlayHidden()
	{
		_peekButton.Disable();
		_skipButton.Disable();
		base.Visible = false;
	}

	private void UpdateControllerIcons()
	{
		_inspectPrompt.Modulate = (NControllerManager.Instance.IsUsingController ? Colors.White : Colors.Transparent);
		_inspectPrompt.GetNode<TextureRect>("ControllerIcon").Texture = NInputManager.Instance.GetHotkeyIcon(MegaInput.accept);
		_inspectPrompt.GetNode<MegaLabel>("Label").SetTextAutoSize(new LocString("gameplay_ui", "TO_INSPECT_PROMPT").GetFormattedText());
	}
}
