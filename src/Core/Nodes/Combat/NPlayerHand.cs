using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Extensions;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NPlayerHand : Control
{
	[Signal]
	public delegate void ModeChangedEventHandler();

	public enum Mode
	{
		None,
		Play,
		SimpleSelect,
		UpgradeSelect
	}

	private StringName[] _selectCardShortcuts = new StringName[10]
	{
		MegaInput.selectCard1,
		MegaInput.selectCard2,
		MegaInput.selectCard3,
		MegaInput.selectCard4,
		MegaInput.selectCard5,
		MegaInput.selectCard6,
		MegaInput.selectCard7,
		MegaInput.selectCard8,
		MegaInput.selectCard9,
		MegaInput.selectCard10
	};

	private Control _selectModeBackstop;

	private readonly List<CardModel> _selectedCards = new List<CardModel>();

	private CardSelectorPrefs _prefs;

	private TaskCompletionSource<IEnumerable<CardModel>>? _selectionCompletionSource;

	private Control _upgradePreviewContainer;

	private NSelectedHandCardContainer _selectedHandCardContainer;

	private NUpgradePreview _upgradePreview;

	private NConfirmButton _selectModeConfirmButton;

	private MegaRichTextLabel _selectionHeader;

	private NCardPlay? _currentCardPlay;

	private CombatState? _combatState;

	private Mode _currentMode = Mode.Play;

	private Func<CardModel, bool>? _currentSelectionFilter;

	private int _draggedHolderIndex = -1;

	private int _lastFocusedHolderIdx = -1;

	private readonly Dictionary<NHandCardHolder, int> _holdersAwaitingQueue = new Dictionary<NHandCardHolder, int>();

	private Tween? _animEnableTween;

	private bool _isDisabled;

	private const double _enableDisableDuration = 0.2;

	private static readonly Vector2 _disablePosition = new Vector2(0f, 100f);

	private static readonly Color _disableModulate = StsColors.gray;

	private Tween? _animInTween;

	private Tween? _animOutTween;

	private Tween? _selectedCardScaleTween;

	private const float _showHideAnimDuration = 0.8f;

	private static readonly Vector2 _showPosition = Vector2.Zero;

	private static readonly Vector2 _hidePosition = new Vector2(0f, 500f);

	public static NPlayerHand? Instance => NCombatRoom.Instance?.Ui.Hand;

	public Control CardHolderContainer { get; private set; }

	public NPeekButton PeekButton { get; private set; }

	public bool InCardPlay
	{
		get
		{
			if (_currentCardPlay != null)
			{
				return GodotObject.IsInstanceValid(_currentCardPlay);
			}
			return false;
		}
	}

	public bool IsInCardSelection
	{
		get
		{
			Mode currentMode = CurrentMode;
			if ((uint)(currentMode - 2) <= 1u)
			{
				return true;
			}
			return false;
		}
	}

	public Mode CurrentMode
	{
		get
		{
			return _currentMode;
		}
		private set
		{
			_currentMode = value;
			EmitSignal(SignalName.ModeChanged);
		}
	}

	private bool HasDraggedHolder => _draggedHolderIndex >= 0;

	public Func<CardModel, bool>? SelectModeGoldGlowOverride => _prefs.ShouldGlowGold;

	public NHandCardHolder? FocusedHolder { get; private set; }

	public IReadOnlyList<NHandCardHolder> ActiveHolders => Holders.Where((NHandCardHolder child) => child.Visible).ToList();

	private IReadOnlyList<NHandCardHolder> Holders => CardHolderContainer.GetChildren().OfType<NHandCardHolder>().ToList();

	public Control DefaultFocusedControl
	{
		get
		{
			if (ActiveHolders.Count > 0)
			{
				if (_lastFocusedHolderIdx >= 0)
				{
					return ActiveHolders[Mathf.Clamp(_lastFocusedHolderIdx, 0, ActiveHolders.Count - 1)];
				}
				return ActiveHolders[ActiveHolders.Count / 2];
			}
			return CardHolderContainer;
		}
	}

	public override void _Ready()
	{
		_selectModeBackstop = GetNode<Control>("%SelectModeBackstop");
		CardHolderContainer = GetNode<Control>("%CardHolderContainer");
		_upgradePreviewContainer = GetNode<Control>("%UpgradePreviewContainer");
		_selectModeConfirmButton = GetNode<NConfirmButton>("%SelectModeConfirmButton");
		_upgradePreview = GetNode<NUpgradePreview>("%UpgradePreview");
		_selectionHeader = GetNode<MegaRichTextLabel>("%SelectionHeader");
		_selectionHeader.Visible = false;
		_selectedHandCardContainer = GetNode<NSelectedHandCardContainer>("%SelectedHandCardContainer");
		_selectedHandCardContainer.Hand = this;
		_selectModeConfirmButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnSelectModeConfirmButtonPressed));
		_selectModeConfirmButton.Disable();
		_selectedHandCardContainer.Connect(Node.SignalName.ChildExitingTree, Callable.From<Node>(OnCardDeselected));
		_selectedHandCardContainer.Connect(Node.SignalName.ChildEnteredTree, Callable.From<Node>(OnCardSelected));
		PeekButton = GetNode<NPeekButton>("%PeekButton");
		PeekButton.Disable();
		PeekButton.AddTargets(_selectModeBackstop, _upgradePreviewContainer, _selectModeConfirmButton, _selectionHeader, _selectedHandCardContainer);
		PeekButton.Connect(NPeekButton.SignalName.Toggled, Callable.From<NPeekButton>(OnPeekButtonToggled));
		CardHolderContainer.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
		{
			if (ActiveHolders.Count > 0)
			{
				DefaultFocusedControl.TryGrabFocus();
			}
		}));
		CardHolderContainer.FocusNeighborBottom = CardHolderContainer.GetPath();
		CardHolderContainer.FocusNeighborLeft = CardHolderContainer.GetPath();
		CardHolderContainer.FocusNeighborRight = CardHolderContainer.GetPath();
	}

	public override void _EnterTree()
	{
		base._EnterTree();
		CombatManager.Instance.PlayerActionsDisabledChanged += OnPlayerActionsDisabledChanged;
		CombatManager.Instance.PlayerUnendedTurn += OnPlayerUnendedTurn;
		CombatManager.Instance.StateTracker.CombatStateChanged += OnCombatStateChanged;
		CombatManager.Instance.CombatEnded += OnCombatEnded;
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		TaskCompletionSource<IEnumerable<CardModel>> selectionCompletionSource = _selectionCompletionSource;
		if (selectionCompletionSource != null)
		{
			Task<IEnumerable<CardModel>> task = selectionCompletionSource.Task;
			if (task != null && !task.IsCompleted)
			{
				_selectionCompletionSource.SetResult(Array.Empty<CardModel>());
			}
		}
		CombatManager.Instance.PlayerActionsDisabledChanged -= OnPlayerActionsDisabledChanged;
		CombatManager.Instance.PlayerUnendedTurn -= OnPlayerUnendedTurn;
		CombatManager.Instance.StateTracker.CombatStateChanged -= OnCombatStateChanged;
		CombatManager.Instance.CombatEnded -= OnCombatEnded;
	}

	public NCard? GetCard(CardModel card)
	{
		return GetCardHolder(card)?.CardNode;
	}

	public bool IsAwaitingPlay(NHandCardHolder? holder)
	{
		if (holder != null)
		{
			return _holdersAwaitingQueue.ContainsKey(holder);
		}
		return false;
	}

	public NCardHolder? GetCardHolder(CardModel card)
	{
		return ((IEnumerable<NCardHolder>)Holders).Concat((IEnumerable<NCardHolder>)_selectedHandCardContainer.Holders).Concat(_holdersAwaitingQueue.Keys).FirstOrDefault((NCardHolder h) => h.CardNode != null && h.CardNode.Model == card);
	}

	public NHandCardHolder Add(NCard card, int index = -1)
	{
		Vector2 globalPosition = card.GlobalPosition;
		NHandCardHolder nHandCardHolder = NHandCardHolder.Create(card, this);
		AddCardHolder(nHandCardHolder, index);
		nHandCardHolder.GlobalPosition = globalPosition;
		RefreshLayout();
		return nHandCardHolder;
	}

	public void Remove(CardModel card)
	{
		NCardHolder cardHolder = GetCardHolder(card);
		if (cardHolder == null)
		{
			throw new InvalidOperationException($"No holder for card {card.Id}");
		}
		if (InCardPlay && card == _currentCardPlay.Holder.CardModel)
		{
			_currentCardPlay.CancelPlayCard();
		}
		RemoveCardHolder(cardHolder);
	}

	private void AddCardHolder(NHandCardHolder holder, int index)
	{
		CardHolderContainer.AddChildSafely(holder);
		if (index >= 0)
		{
			CardHolderContainer.MoveChild(holder, index);
		}
		holder.Connect(NCardHolder.SignalName.Pressed, Callable.From<NCardHolder>(OnHolderPressed));
		holder.Connect(NHandCardHolder.SignalName.HolderMouseClicked, Callable.From<NCardHolder>(OnHolderPressed));
		holder.Connect(NHandCardHolder.SignalName.HolderFocused, Callable.From<NHandCardHolder>(OnHolderFocused));
		holder.Connect(NHandCardHolder.SignalName.HolderUnfocused, Callable.From<NHandCardHolder>(OnHolderUnfocused));
		RefreshLayout();
		if (CardHolderContainer.HasFocus())
		{
			holder.TryGrabFocus();
		}
	}

	public void RemoveCardHolder(NCardHolder holder)
	{
		if (holder is NHandCardHolder key)
		{
			_holdersAwaitingQueue.Remove(key);
		}
		if (InCardPlay && _currentCardPlay.Holder == holder)
		{
			_currentCardPlay.CancelPlayCard();
		}
		bool flag = holder.HasFocus();
		holder.Clear();
		holder.GetParent().RemoveChildSafely(holder);
		holder.QueueFreeSafely();
		RefreshLayout();
		if (flag)
		{
			DefaultFocusedControl.TryGrabFocus();
		}
	}

	private void OnHolderFocused(NHandCardHolder holder)
	{
		FocusedHolder = holder;
		_lastFocusedHolderIdx = holder.GetIndex();
		RunManager.Instance.HoveredModelTracker.OnLocalCardHovered(FocusedHolder.CardModel);
		RefreshLayout();
	}

	private void OnHolderUnfocused(NHandCardHolder holder)
	{
		FocusedHolder = null;
		RunManager.Instance.HoveredModelTracker.OnLocalCardUnhovered();
		RefreshLayout();
	}

	public void TryCancelCardPlay(CardModel card)
	{
		NCardHolder cardHolder = GetCardHolder(card);
		if (cardHolder is NHandCardHolder nHandCardHolder && IsAwaitingPlay(nHandCardHolder))
		{
			ReturnHolderToHand(nHandCardHolder);
			nHandCardHolder.UpdateCard();
			if (InCardPlay && _currentCardPlay.Holder == nHandCardHolder)
			{
				_currentCardPlay.CancelPlayCard();
			}
			else
			{
				RefreshLayout();
			}
		}
	}

	public void CancelAllCardPlay()
	{
		if (InCardPlay)
		{
			_currentCardPlay.CancelPlayCard();
		}
		foreach (NHandCardHolder item in _holdersAwaitingQueue.Keys.ToList())
		{
			ReturnHolderToHand(item);
		}
	}

	private void ReturnHolderToHand(NHandCardHolder holder)
	{
		if (IsAwaitingPlay(holder))
		{
			int num = _holdersAwaitingQueue[holder];
			_holdersAwaitingQueue.Remove(holder);
			holder.Reparent(CardHolderContainer);
			if (num >= 0)
			{
				CardHolderContainer.MoveChild(holder, num);
			}
			holder.SetDefaultTargets();
		}
	}

	public void ForceRefreshCardIndices()
	{
		RefreshLayout();
	}

	private void RefreshLayout()
	{
		int count = ActiveHolders.Count;
		if (count <= 0)
		{
			return;
		}
		int handSize = count;
		Vector2 scale = HandPosHelper.GetScale(count);
		int num = -1;
		if (FocusedHolder != null)
		{
			num = ActiveHolders.IndexOf<NHandCardHolder>(FocusedHolder);
		}
		for (int i = 0; i < count; i++)
		{
			int cardIndex = i;
			Vector2 position = HandPosHelper.GetPosition(handSize, cardIndex);
			if (num > -1)
			{
				float num2 = Mathf.Lerp(100f, 0f, Mathf.Min(1f, (float)Mathf.Abs(num - i) / 4f));
				position += Vector2.Left * Mathf.Sign(num - i) * num2;
			}
			NHandCardHolder nHandCardHolder = ActiveHolders[i];
			if (num == i)
			{
				nHandCardHolder.SetAngleInstantly(0f);
				nHandCardHolder.SetScaleInstantly(Vector2.One);
				position.Y = (0f - nHandCardHolder.Hitbox.Size.Y) * 0.5f + 2f;
				if (_isDisabled)
				{
					position -= _disablePosition;
				}
				nHandCardHolder.Position = new Vector2(nHandCardHolder.Position.X, position.Y);
				nHandCardHolder.SetTargetPosition(position);
			}
			else
			{
				nHandCardHolder.SetTargetPosition(position);
				nHandCardHolder.SetTargetScale(scale);
				nHandCardHolder.SetTargetAngle(HandPosHelper.GetAngle(handSize, cardIndex));
			}
			nHandCardHolder.Hitbox.MouseFilter = (MouseFilterEnum)(HasDraggedHolder ? 2 : 0);
			NodePath path;
			if (i <= 0)
			{
				IReadOnlyList<NHandCardHolder> activeHolders = ActiveHolders;
				path = activeHolders[activeHolders.Count - 1].GetPath();
			}
			else
			{
				path = ActiveHolders[i - 1].GetPath();
			}
			nHandCardHolder.FocusNeighborLeft = path;
			nHandCardHolder.FocusNeighborRight = ((i < ActiveHolders.Count - 1) ? ActiveHolders[i + 1].GetPath() : ActiveHolders[0].GetPath());
			nHandCardHolder.FocusNeighborBottom = nHandCardHolder.GetPath();
			if (HasDraggedHolder && i >= _draggedHolderIndex)
			{
				nHandCardHolder.SetIndexLabel(i + 2);
			}
			else
			{
				nHandCardHolder.SetIndexLabel(i + 1);
			}
		}
	}

	private void OnPlayerUnendedTurn(Player player)
	{
		UpdateHandDisabledState(player.Creature.CombatState);
	}

	private void OnPlayerActionsDisabledChanged(CombatState state)
	{
		UpdateHandDisabledState(state);
	}

	private void UpdateHandDisabledState(CombatState state)
	{
		Player me = LocalContext.GetMe(state);
		bool flag = CombatManager.Instance.PlayerActionsDisabled;
		if (!flag && CombatManager.Instance.PlayersTakingExtraTurn.Count > 0 && me != null && !CombatManager.Instance.PlayersTakingExtraTurn.Contains(me))
		{
			flag = true;
		}
		if (flag)
		{
			if (me == null || !state.Players.Except(new global::_003C_003Ez__ReadOnlySingleElementList<Player>(me)).All(CombatManager.Instance.IsPlayerReadyToEndTurn))
			{
				AnimDisable();
			}
		}
		else
		{
			AnimEnable();
		}
	}

	private void OnCombatStateChanged(CombatState state)
	{
		_combatState = state;
		foreach (NHandCardHolder holder in Holders)
		{
			holder.UpdateCard();
		}
		foreach (NHandCardHolder key in _holdersAwaitingQueue.Keys)
		{
			key.UpdateCard();
		}
		foreach (NSelectedHandCardHolder holder2 in _selectedHandCardContainer.Holders)
		{
			holder2.CardNode?.UpdateVisuals(PileType.Hand, CardPreviewMode.Normal);
		}
		UpdateHandDisabledState(state);
	}

	private void OnCombatEnded(CombatRoom _)
	{
		CancelAllCardPlay();
		CardHolderContainer.FocusMode = FocusModeEnum.None;
	}

	private void OnPeekButtonToggled(NPeekButton button)
	{
		if (button.IsPeeking)
		{
			NCombatRoom.Instance.EnableControllerNavigation();
		}
		else
		{
			NCombatRoom.Instance.RestrictControllerNavigation(Array.Empty<Control>());
			EnableControllerNavigation();
		}
		UpdateSelectModeCardVisibility();
		ActiveScreenContext.Instance.Update();
	}

	public async Task<IEnumerable<CardModel>> SelectCards(CardSelectorPrefs prefs, Func<CardModel, bool>? filter, AbstractModel? source, Mode mode = Mode.SimpleSelect)
	{
		CancelAllCardPlay();
		_selectModeBackstop.Visible = true;
		_selectModeBackstop.MouseFilter = MouseFilterEnum.Stop;
		Control selectModeBackstop = _selectModeBackstop;
		Color selfModulate = _selectModeBackstop.SelfModulate;
		selfModulate.A = 0f;
		selectModeBackstop.SelfModulate = selfModulate;
		Tween tween = CreateTween();
		tween.TweenProperty(_selectModeBackstop, "self_modulate:a", 1f, 0.20000000298023224);
		bool wasDisabled = _isDisabled;
		if (_isDisabled)
		{
			AnimEnable();
		}
		CurrentMode = mode;
		_currentSelectionFilter = filter;
		NCombatRoom.Instance.RestrictControllerNavigation(Array.Empty<Control>());
		NCombatUi ui = NCombatRoom.Instance.Ui;
		ui.OnHandSelectModeEntered();
		EnableControllerNavigation();
		_prefs = prefs;
		_selectionCompletionSource = new TaskCompletionSource<IEnumerable<CardModel>>();
		_selectionHeader.Visible = true;
		_selectionHeader.Text = "[center]" + prefs.Prompt.GetFormattedText() + "[/center]";
		PeekButton.Enable();
		UpdateSelectModeCardVisibility();
		RefreshSelectModeConfirmButton();
		IEnumerable<CardModel> result = await _selectionCompletionSource.Task;
		tween.Kill();
		AfterCardsSelected(source);
		if (wasDisabled)
		{
			AnimDisable();
		}
		return result;
	}

	private void UpdateSelectModeCardVisibility()
	{
		if (CurrentMode != Mode.SimpleSelect && CurrentMode != Mode.UpgradeSelect)
		{
			throw new InvalidOperationException("Can only be used when we are selecting a card");
		}
		foreach (NHandCardHolder holder in Holders)
		{
			if (holder.CardNode != null)
			{
				if (PeekButton.IsPeeking)
				{
					holder.Visible = true;
					holder.CardNode.SetPretendCardCanBePlayed(pretendCardCanBePlayed: false);
					holder.CardNode.SetForceUnpoweredPreview(forceUnpoweredPreview: false);
				}
				else
				{
					holder.Visible = _currentSelectionFilter?.Invoke(holder.CardNode.Model) ?? true;
					holder.CardNode.SetPretendCardCanBePlayed(_prefs.PretendCardsCanBePlayed);
					holder.CardNode.SetForceUnpoweredPreview(_prefs.UnpoweredPreviews);
				}
				holder.UpdateCard();
			}
		}
		RefreshLayout();
	}

	private void AfterCardsSelected(AbstractModel? source)
	{
		_selectedCards.Clear();
		foreach (NHandCardHolder holder in Holders)
		{
			holder.InSelectMode = false;
			holder.Visible = true;
			holder.CardNode?.SetPretendCardCanBePlayed(pretendCardCanBePlayed: false);
			holder.CardNode?.SetForceUnpoweredPreview(forceUnpoweredPreview: false);
			holder.UpdateCard();
		}
		RefreshLayout();
		_selectModeBackstop.Visible = false;
		_selectModeBackstop.MouseFilter = MouseFilterEnum.Ignore;
		Tween tween = CreateTween();
		tween.TweenProperty(_selectModeBackstop, "self_modulate:a", 0f, 0.20000000298023224);
		_selectModeConfirmButton.Disable();
		_upgradePreviewContainer.Visible = false;
		_selectionHeader.Visible = false;
		PeekButton.Disable();
		_prefs = default(CardSelectorPrefs);
		CurrentMode = Mode.Play;
		_currentSelectionFilter = null;
		NCombatRoom.Instance.Ui.OnHandSelectModeExited();
		if (source != null)
		{
			source.ExecutionFinished += OnSelectModeSourceFinished;
		}
		else
		{
			OnSelectModeSourceFinished(null);
		}
	}

	private void CancelHandSelectionIfNecessary()
	{
		if (IsInCardSelection && _selectionCompletionSource != null)
		{
			_selectionCompletionSource.SetCanceled();
			AfterCardsSelected(null);
		}
	}

	private void OnHolderPressed(NCardHolder holder)
	{
		if (PeekButton.IsPeeking)
		{
			PeekButton.Wiggle();
			return;
		}
		NHandCardHolder nHandCardHolder = (NHandCardHolder)holder;
		if (nHandCardHolder.CardNode == null || !CombatManager.Instance.IsInProgress || NOverlayStack.Instance.ScreenCount > 0)
		{
			return;
		}
		switch (CurrentMode)
		{
		case Mode.Play:
			if (CanPlayCards())
			{
				StartCardPlay(nHandCardHolder, startedViaShortcut: false);
			}
			break;
		case Mode.SimpleSelect:
			SelectCardInSimpleMode(nHandCardHolder);
			break;
		case Mode.UpgradeSelect:
			SelectCardInUpgradeMode(nHandCardHolder);
			break;
		default:
			throw new ArgumentOutOfRangeException("CurrentMode");
		case Mode.None:
			break;
		}
	}

	private bool CanPlayCards()
	{
		if (!InCardPlay)
		{
			return AreCardActionsAllowed();
		}
		return false;
	}

	private bool AreCardActionsAllowed()
	{
		if (CombatManager.Instance.PlayersTakingExtraTurn.Count > 0 && _combatState != null)
		{
			Player me = LocalContext.GetMe(_combatState);
			if (me == null || !CombatManager.Instance.PlayersTakingExtraTurn.Contains(me))
			{
				return false;
			}
		}
		if (!CombatManager.Instance.PlayerActionsDisabled)
		{
			return !PeekButton.IsPeeking;
		}
		return false;
	}

	private void StartCardPlay(NHandCardHolder holder, bool startedViaShortcut)
	{
		_draggedHolderIndex = holder.GetIndex();
		_holdersAwaitingQueue.Add(holder, _draggedHolderIndex);
		holder.Reparent(this);
		holder.BeginDrag();
		_currentCardPlay = (NControllerManager.Instance.IsUsingController ? ((NCardPlay)NControllerCardPlay.Create(holder)) : ((NCardPlay)NMouseCardPlay.Create(holder, _selectCardShortcuts[_draggedHolderIndex], startedViaShortcut)));
		this.AddChildSafely(_currentCardPlay);
		_currentCardPlay.Connect(NCardPlay.SignalName.Finished, Callable.From(delegate(bool success)
		{
			RunManager.Instance.HoveredModelTracker.OnLocalCardDeselected();
			if (!success)
			{
				ReturnHolderToHand(holder);
			}
			_draggedHolderIndex = -1;
			RefreshLayout();
		}));
		RunManager.Instance.HoveredModelTracker.OnLocalCardSelected(holder.CardNode.Model);
		_currentCardPlay.Start();
		RefreshLayout();
		holder.SetIndexLabel(_draggedHolderIndex + 1);
	}

	private void SelectCardInSimpleMode(NHandCardHolder holder)
	{
		if (_selectedCards.Count >= _prefs.MaxSelect)
		{
			_selectedHandCardContainer.DeselectCard(_selectedCards.Last());
		}
		_selectedCards.Add(holder.CardNode.Model);
		_selectedHandCardContainer.Add(holder);
		RemoveCardHolder(holder);
		RefreshSelectModeConfirmButton();
	}

	private void SelectCardInUpgradeMode(NHandCardHolder holder)
	{
		CardModel model = holder.CardNode.Model;
		if (_selectedCards.Count != 0)
		{
			NCard nCard = NCard.Create(_selectedCards.Last());
			nCard.GlobalPosition = _upgradePreview.SelectedCardPosition;
			DeselectCard(nCard);
		}
		_selectedCards.Add(model);
		_upgradePreviewContainer.Visible = true;
		_upgradePreview.Card = model;
		RemoveCardHolder(holder);
		RefreshSelectModeConfirmButton();
	}

	public void DeselectCard(NCard card)
	{
		if (!IsInCardSelection)
		{
			throw new InvalidOperationException("Only valid when in Select Mode.");
		}
		NHandCardHolder nHandCardHolder = Add(card, PileType.Hand.GetPile(card.Model.Owner).Cards.IndexOf<CardModel>(card.Model));
		nHandCardHolder.InSelectMode = true;
		nHandCardHolder.Visible = true;
		_selectedCards.Remove(card.Model);
		RefreshSelectModeConfirmButton();
		nHandCardHolder.TryGrabFocus();
	}

	private void OnSelectModeConfirmButtonPressed(NButton _)
	{
		_selectionCompletionSource.SetResult(_selectedCards.ToList());
	}

	private void CheckIfSelectionComplete()
	{
		if (_selectedCards.Count >= _prefs.MaxSelect)
		{
			_selectionCompletionSource.SetResult(_selectedCards.ToList());
		}
	}

	private void RefreshSelectModeConfirmButton()
	{
		int count = _selectedCards.Count;
		if (count >= _prefs.MinSelect && count <= _prefs.MaxSelect)
		{
			_selectModeConfirmButton.Enable();
		}
		else
		{
			_selectModeConfirmButton.Disable();
		}
	}

	private void OnSelectModeSourceFinished(AbstractModel? source)
	{
		foreach (NSelectedHandCardHolder item in _selectedHandCardContainer.Holders.ToList())
		{
			NCard cardNode = item.CardNode;
			item.QueueFreeSafely();
			Add(cardNode);
		}
		if (_upgradePreview.Card != null)
		{
			Add(NCard.Create(_upgradePreview.Card));
			_upgradePreview.Card = null;
		}
		if (source != null)
		{
			source.ExecutionFinished -= OnSelectModeSourceFinished;
		}
	}

	public void AnimIn()
	{
		_animOutTween?.Kill();
		_animEnableTween?.Kill();
		_animInTween = CreateTween();
		_animInTween.TweenProperty(this, "position", _showPosition, 0.800000011920929).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	public void AnimOut()
	{
		CancelHandSelectionIfNecessary();
		_animInTween?.Kill();
		_animEnableTween?.Kill();
		_animOutTween = CreateTween();
		_animOutTween.TweenProperty(this, "position", _hidePosition, 0.800000011920929).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Back);
	}

	private void AnimDisable()
	{
		if (!_isDisabled)
		{
			DisableControllerNavigation();
			_animEnableTween = CreateTween().SetParallel();
			_animEnableTween.TweenProperty(this, "position", _disablePosition, 0.2).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
			_animEnableTween.TweenProperty(this, "modulate", _disableModulate, 0.2).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
			_isDisabled = true;
		}
	}

	private void AnimEnable()
	{
		if (_isDisabled)
		{
			EnableControllerNavigation();
			DefaultFocusedControl.TryGrabFocus();
			_animEnableTween = CreateTween().SetParallel();
			_animEnableTween.TweenProperty(this, "position", _showPosition, 0.2).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
			_animEnableTween.TweenProperty(this, "modulate", Colors.White, 0.2).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
			_isDisabled = false;
		}
	}

	public void FlashPlayableHolders()
	{
		foreach (NHandCardHolder holder in Holders)
		{
			if (holder.CardNode != null && holder.CardNode.Model.CanPlay())
			{
				holder.Flash();
			}
		}
	}

	private void OnCardSelected(Node _)
	{
		UpdateSelectedCardContainer(_selectedHandCardContainer.GetChildCount());
	}

	private void OnCardDeselected(Node _)
	{
		UpdateSelectedCardContainer(_selectedHandCardContainer.GetChildCount() - 1);
	}

	private void UpdateSelectedCardContainer(int count)
	{
		float num = 1f;
		float num2 = base.Size.Y * 0.5f;
		if (count > 6)
		{
			num = 0.55f;
			num2 -= 150f;
		}
		else if (count > 3)
		{
			num = 0.8f;
			num2 -= 75f;
		}
		_selectedCardScaleTween?.Kill();
		_selectedCardScaleTween = CreateTween().SetParallel();
		_selectedCardScaleTween.TweenProperty(_selectedHandCardContainer, "position:y", num2, 0.5).SetTrans(Tween.TransitionType.Quad).SetEase(Tween.EaseType.InOut);
		_selectedCardScaleTween.TweenProperty(_selectedHandCardContainer, "scale", Vector2.One * num, 0.5).SetTrans(Tween.TransitionType.Quad).SetEase(Tween.EaseType.InOut);
	}

	public void EnableControllerNavigation()
	{
		foreach (NHandCardHolder holder in Holders)
		{
			holder.FocusMode = FocusModeEnum.All;
		}
		if (InCardPlay)
		{
			_currentCardPlay.Holder.FocusMode = FocusModeEnum.All;
		}
	}

	public void DisableControllerNavigation()
	{
		foreach (NHandCardHolder holder in Holders)
		{
			holder.FocusMode = FocusModeEnum.None;
		}
		if (InCardPlay)
		{
			_currentCardPlay.Holder.FocusMode = FocusModeEnum.None;
		}
	}

	public override void _UnhandledInput(InputEvent input)
	{
		if (NControllerManager.Instance.IsUsingController || !ActiveScreenContext.Instance.IsCurrent(NCombatRoom.Instance) || CombatManager.Instance.IsOverOrEnding)
		{
			return;
		}
		List<NHandCardHolder> list = new List<NHandCardHolder>();
		list.AddRange(ActiveHolders);
		if (HasDraggedHolder)
		{
			list.Insert(_draggedHolderIndex, null);
		}
		for (int i = 0; i < _selectCardShortcuts.Length; i++)
		{
			StringName action = _selectCardShortcuts[i];
			if (!input.IsActionPressed(action) || list.Count <= i)
			{
				continue;
			}
			NHandCardHolder nHandCardHolder = list[i];
			if (nHandCardHolder == null)
			{
				continue;
			}
			if (NTargetManager.Instance.IsInSelection)
			{
				NTargetManager.Instance.CancelTargeting();
			}
			switch (CurrentMode)
			{
			case Mode.Play:
				if (AreCardActionsAllowed())
				{
					if (InCardPlay)
					{
						_currentCardPlay.CancelPlayCard();
					}
					StartCardPlay(nHandCardHolder, startedViaShortcut: true);
				}
				break;
			case Mode.SimpleSelect:
				if (!PeekButton.IsPeeking)
				{
					SelectCardInSimpleMode(nHandCardHolder);
				}
				break;
			case Mode.UpgradeSelect:
				if (!PeekButton.IsPeeking)
				{
					SelectCardInUpgradeMode(nHandCardHolder);
				}
				break;
			}
			GetViewport()?.SetInputAsHandled();
		}
	}
}
