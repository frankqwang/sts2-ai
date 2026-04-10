using System;
using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens;

public partial class NInspectCardScreen : Control, IScreenContext
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/inspect_card_screen");

	private NCard _card;

	private NButton _backstop;

	private NTickbox _upgradeTickbox;

	private NButton _leftButton;

	private NButton _rightButton;

	private Control _hoverTipRect;

	private List<CardModel>? _cards;

	private int _index;

	private Tween? _openTween;

	private Tween? _cardTween;

	private Vector2 _cardPosition;

	private float _leftButtonX;

	private float _rightButtonX;

	private const double _arrowButtonDelay = 0.1;

	private bool _viewAllUpgraded;

	public static string[] AssetPaths => new string[1] { _scenePath };

	private bool IsShowingUpgradedCard => _upgradeTickbox.IsTicked;

	public Control? DefaultFocusedControl => null;

	public static NInspectCardScreen? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NInspectCardScreen>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		_card = GetNode<NCard>("Card");
		_cardPosition = _card.Position;
		_hoverTipRect = GetNode<Control>("HoverTipRect");
		_backstop = GetNode<NButton>("Backstop");
		_backstop.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnBackstopPressed));
		_leftButton = GetNode<NButton>("LeftArrow");
		_leftButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(delegate
		{
			OnLeftButtonReleased();
		}));
		_rightButton = GetNode<NButton>("RightArrow");
		_rightButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(delegate
		{
			OnRightButtonReleased();
		}));
		_leftButtonX = _leftButton.Position.X;
		_rightButtonX = _rightButton.Position.X;
		_upgradeTickbox = GetNode<NTickbox>("%Upgrade");
		_upgradeTickbox.IsTicked = false;
		_upgradeTickbox.Connect(NTickbox.SignalName.Toggled, Callable.From<NTickbox>(ToggleShowUpgrade));
		GetNode<MegaLabel>("%ShowUpgradeLabel").SetTextAutoSize(new LocString("card_selection", "VIEW_UPGRADES").GetFormattedText());
		_rightButton.Disable();
		_leftButton.Disable();
		_upgradeTickbox.Disable();
		Close();
	}

	public void Open(List<CardModel> cards, int index, bool viewAllUpgraded = false)
	{
		_cards = cards;
		base.Visible = true;
		base.MouseFilter = MouseFilterEnum.Stop;
		_viewAllUpgraded = viewAllUpgraded;
		SetCard(index);
		_card.Scale = Vector2.One * 1.75f;
		_card.Modulate = StsColors.transparentBlack;
		_leftButton.Modulate = StsColors.transparentBlack;
		_rightButton.Modulate = StsColors.transparentBlack;
		_rightButton.Enable();
		_leftButton.Enable();
		_openTween?.Kill();
		_openTween = CreateTween().SetParallel();
		_openTween.TweenProperty(_backstop, "modulate:a", 0.9f, 0.25);
		_openTween.TweenProperty(this, "modulate:a", 1f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
			.From(0f);
		_openTween.TweenProperty(_leftButton, "position:x", _leftButtonX, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back)
			.From(_leftButtonX + 100f)
			.SetDelay(0.1);
		_openTween.TweenProperty(_leftButton, "modulate", Colors.White, 0.25).SetDelay(0.1);
		_openTween.TweenProperty(_rightButton, "position:x", _rightButtonX, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back)
			.From(_rightButtonX - 100f)
			.SetDelay(0.1);
		_openTween.TweenProperty(_rightButton, "modulate", Colors.White, 0.25).SetDelay(0.1);
		_cardTween?.Kill();
		_cardTween = CreateTween().SetParallel();
		_cardTween.TweenProperty(_card, "modulate", Colors.White, 0.25);
		_cardTween.TweenProperty(_card, "scale", Vector2.One * 2f, 0.15).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Spring)
			.SetDelay(0.1);
		_upgradeTickbox.Enable();
		ActiveScreenContext.Instance.Update();
		NHotkeyManager.Instance.AddBlockingScreen(this);
		NHotkeyManager.Instance.PushHotkeyPressedBinding(MegaInput.cancel, Close);
		NHotkeyManager.Instance.PushHotkeyPressedBinding(MegaInput.pauseAndBack, Close);
		NHotkeyManager.Instance.PushHotkeyPressedBinding(MegaInput.left, OnLeftButtonReleased);
		NHotkeyManager.Instance.PushHotkeyPressedBinding(MegaInput.right, OnRightButtonReleased);
	}

	public void Close()
	{
		if (base.Visible)
		{
			base.MouseFilter = MouseFilterEnum.Ignore;
			_leftButton.MouseFilter = MouseFilterEnum.Ignore;
			_rightButton.MouseFilter = MouseFilterEnum.Ignore;
			_rightButton.Disable();
			_leftButton.Disable();
			_upgradeTickbox.Disable();
			NHoverTipSet.Clear();
			SetProcessUnhandledInput(enable: false);
			_openTween?.Kill();
			_openTween = CreateTween().SetParallel();
			_openTween.TweenProperty(_backstop, "modulate:a", 0f, 0.25);
			_openTween.TweenProperty(_leftButton, "modulate:a", 0f, 0.1);
			_openTween.TweenProperty(_rightButton, "modulate:a", 0f, 0.1);
			_openTween.TweenProperty(_card, "modulate", StsColors.transparentWhite, 0.1);
			_openTween.Chain().TweenCallback(Callable.From(delegate
			{
				base.Visible = false;
				ActiveScreenContext.Instance.Update();
			}));
			NHotkeyManager.Instance.RemoveHotkeyPressedBinding(MegaInput.cancel, Close);
			NHotkeyManager.Instance.RemoveHotkeyPressedBinding(MegaInput.pauseAndBack, Close);
			NHotkeyManager.Instance.RemoveHotkeyPressedBinding(MegaInput.left, OnLeftButtonReleased);
			NHotkeyManager.Instance.RemoveHotkeyPressedBinding(MegaInput.right, OnRightButtonReleased);
			NHotkeyManager.Instance.RemoveBlockingScreen(this);
		}
	}

	private void OnRightButtonReleased()
	{
		if (_rightButton.Visible)
		{
			SetCard(_index + 1);
			_card.Modulate = Colors.White;
			_openTween?.Kill();
			_openTween = CreateTween().SetParallel();
			_openTween.TweenProperty(_card, "position", _cardPosition, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
				.From(_cardPosition + new Vector2(100f, 0f));
		}
	}

	private void OnLeftButtonReleased()
	{
		if (_leftButton.Visible)
		{
			SetCard(_index - 1);
			_card.Modulate = Colors.White;
			_openTween?.Kill();
			_openTween = CreateTween().SetParallel();
			_openTween.TweenProperty(_card, "position", _cardPosition, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
				.From(_cardPosition + new Vector2(-100f, 0f));
		}
	}

	private void ToggleShowUpgrade(NTickbox _)
	{
		_viewAllUpgraded = false;
		UpdateCardDisplay();
	}

	private void UpdateCardDisplay()
	{
		CardModel cardModel = _cards[_index];
		CardModel cardModel2 = (CardModel)_cards[_index].MutableClone();
		if (IsShowingUpgradedCard)
		{
			if (!cardModel.IsUpgraded && cardModel.IsUpgradable)
			{
				cardModel2.UpgradePreviewType = CardUpgradePreviewType.Deck;
				cardModel2.UpgradeInternal();
			}
			_card.Model = cardModel2;
			_card.ShowUpgradePreview();
		}
		else
		{
			if (cardModel2.IsUpgraded)
			{
				CardCmd.Downgrade(cardModel2);
			}
			_card.Model = cardModel2;
			_card.UpdateVisuals(PileType.None, CardPreviewMode.Normal);
		}
		NHoverTipSet.Clear();
		NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, cardModel2.HoverTips);
		nHoverTipSet.SetAlignment(_hoverTipRect, HoverTip.GetHoverTipAlignment(this));
	}

	private void SetCard(int index)
	{
		_index = Math.Clamp(index, 0, _cards.Count - 1);
		_leftButton.Visible = _index > 0;
		_leftButton.MouseFilter = (MouseFilterEnum)(_leftButton.Visible ? 0 : 2);
		_rightButton.Visible = _index < _cards.Count - 1;
		_rightButton.MouseFilter = (MouseFilterEnum)(_rightButton.Visible ? 0 : 2);
		_upgradeTickbox.Visible = _cards[_index].MaxUpgradeLevel > 0;
		_upgradeTickbox.MouseFilter = (MouseFilterEnum)((_cards[_index].MaxUpgradeLevel > 0) ? 0 : 2);
		if (_cards[_index].IsUpgraded || _viewAllUpgraded)
		{
			_upgradeTickbox.IsTicked = true;
		}
		else
		{
			_upgradeTickbox.IsTicked = false;
		}
		UpdateCardDisplay();
	}

	private void OnBackstopPressed(NButton _)
	{
		Close();
	}
}
