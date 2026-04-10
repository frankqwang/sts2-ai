using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
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
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Nodes.Vfx;

namespace MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;

public partial class NChooseABundleSelectionScreen : Control, IOverlayScreen, IScreenContext
{
	private Control _bundleRow;

	private IReadOnlyList<IReadOnlyList<CardModel>> _bundles;

	private Control _bundlePreviewContainer;

	private Control _bundlePreviewCards;

	private NBackButton _previewCancelButton;

	private NConfirmButton _previewConfirmButton;

	private NCardBundle? _selectedBundle;

	private NCommonBanner _banner;

	private readonly TaskCompletionSource<IEnumerable<IReadOnlyList<CardModel>>> _completionSource = new TaskCompletionSource<IEnumerable<IReadOnlyList<CardModel>>>();

	private NPeekButton _peekButton;

	private Tween? _fadeTween;

	private const float _cardXSpacing = 400f;

	private Tween? _cardTween;

	private static string ScenePath => SceneHelper.GetScenePath("/screens/card_selection/choose_a_bundle_selection_screen");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	public NetScreenType ScreenType => NetScreenType.CardSelection;

	public bool UseSharedBackstop => true;

	public Control DefaultFocusedControl
	{
		get
		{
			if (!_bundlePreviewContainer.Visible)
			{
				return _bundleRow.GetChild<NCardBundle>(0).Hitbox;
			}
			return _bundlePreviewCards.GetChild<Control>(_bundlePreviewCards.GetChildCount() - 1);
		}
	}

	public override void _Ready()
	{
		_bundleRow = GetNode<Control>("%BundleRow");
		_bundlePreviewContainer = GetNode<Control>("%BundlePreviewContainer");
		_bundlePreviewCards = GetNode<Control>("%Cards");
		_previewCancelButton = GetNode<NBackButton>("%Cancel");
		_previewConfirmButton = GetNode<NConfirmButton>("%Confirm");
		_banner = GetNode<NCommonBanner>("Banner");
		_banner.label.SetTextAutoSize(new LocString("gameplay_ui", "CHOOSE_A_PACK").GetRawText());
		_banner.AnimateIn();
		_previewCancelButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(CancelSelection));
		_previewConfirmButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(ConfirmSelection));
		_previewCancelButton.Disable();
		_previewConfirmButton.Disable();
		Vector2 vector = Vector2.Left * (_bundles.Count - 1) * 400f * 0.5f;
		for (int i = 0; i < _bundles.Count; i++)
		{
			NCardBundle nCardBundle = NCardBundle.Create(_bundles[i]);
			_bundleRow.AddChildSafely(nCardBundle);
			nCardBundle.Connect(NCardBundle.SignalName.Clicked, Callable.From<NCardBundle>(OnBundleClicked));
			nCardBundle.Scale = nCardBundle.smallScale;
			nCardBundle.Position += vector + Vector2.Right * 400f * i;
		}
		for (int j = 0; j < _bundleRow.GetChildCount(); j++)
		{
			NCardBundle child = _bundleRow.GetChild<NCardBundle>(j);
			child.Hitbox.FocusNeighborLeft = ((j > 0) ? _bundleRow.GetChild<NCardBundle>(j - 1).Hitbox.GetPath() : _bundleRow.GetChild<NCardBundle>(_bundleRow.GetChildCount() - 1).Hitbox.GetPath());
			child.Hitbox.FocusNeighborRight = ((j < _bundleRow.GetChildCount() - 1) ? _bundleRow.GetChild<NCardBundle>(j + 1).Hitbox.GetPath() : _bundleRow.GetChild<NCardBundle>(0).Hitbox.GetPath());
			child.Hitbox.FocusNeighborTop = child.Hitbox.GetPath();
			child.Hitbox.FocusNeighborBottom = child.Hitbox.GetPath();
		}
		_bundlePreviewContainer.Visible = false;
		_bundlePreviewContainer.MouseFilter = MouseFilterEnum.Ignore;
		_peekButton = GetNode<NPeekButton>("%PeekButton");
		_peekButton.AddTargets(_banner, _bundleRow, _bundlePreviewContainer);
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		if (!_completionSource.Task.IsCompleted)
		{
			_completionSource.SetCanceled();
		}
	}

	public static NChooseABundleSelectionScreen ShowScreen(IReadOnlyList<IReadOnlyList<CardModel>> bundles)
	{
		NChooseABundleSelectionScreen nChooseABundleSelectionScreen = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NChooseABundleSelectionScreen>(PackedScene.GenEditState.Disabled);
		nChooseABundleSelectionScreen.Name = "NChooseABundleSelectionScreen";
		nChooseABundleSelectionScreen._bundles = bundles;
		NOverlayStack.Instance.Push(nChooseABundleSelectionScreen);
		return nChooseABundleSelectionScreen;
	}

	private void OnBundleClicked(NCardBundle bundleNode)
	{
		_banner.AnimateOut();
		_selectedBundle = bundleNode;
		_bundlePreviewContainer.Visible = true;
		_bundlePreviewContainer.MouseFilter = MouseFilterEnum.Stop;
		_bundleRow.Visible = false;
		_previewCancelButton.Enable();
		_previewConfirmButton.Enable();
		Vector2 vector = Vector2.Right * (bundleNode.Bundle.Count - 1) * 400f * 0.5f;
		IReadOnlyList<NCard> readOnlyList = bundleNode.RemoveCardNodes();
		_cardTween?.Kill();
		_cardTween = CreateTween().SetParallel();
		for (int i = 0; i < readOnlyList.Count; i++)
		{
			Vector2 globalPosition = readOnlyList[i].GlobalPosition;
			NPreviewCardHolder nPreviewCardHolder = NPreviewCardHolder.Create(readOnlyList[i], showHoverTips: true, scaleOnHover: true);
			_bundlePreviewCards.AddChildSafely(nPreviewCardHolder);
			nPreviewCardHolder.GlobalPosition = globalPosition;
			nPreviewCardHolder.Connect(NCardHolder.SignalName.Pressed, Callable.From<NCardHolder>(OpenPreviewScreen));
			readOnlyList[i].UpdateVisuals(PileType.None, CardPreviewMode.Normal);
			_cardTween.TweenProperty(nPreviewCardHolder, "position", vector + Vector2.Left * 400f * i, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		}
		for (int j = 0; j < _bundlePreviewCards.GetChildCount(); j++)
		{
			NPreviewCardHolder child = _bundlePreviewCards.GetChild<NPreviewCardHolder>(j);
			child.FocusNeighborLeft = ((j < _bundlePreviewCards.GetChildCount() - 1) ? _bundlePreviewCards.GetChild(j + 1).GetPath() : _bundlePreviewCards.GetChild(0).GetPath());
			child.FocusNeighborRight = ((j > 0) ? _bundlePreviewCards.GetChild(j - 1).GetPath() : _bundlePreviewCards.GetChild(_bundlePreviewCards.GetChildCount() - 1).GetPath());
			child.FocusNeighborTop = child.Hitbox.GetPath();
			child.FocusNeighborBottom = child.Hitbox.GetPath();
		}
		_bundlePreviewCards.GetChild<Control>(_bundlePreviewCards.GetChildCount() - 1).TryGrabFocus();
	}

	private void OpenPreviewScreen(NCardHolder cardHolder)
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

	private void CancelSelection(NButton _)
	{
		_banner.AnimateIn();
		_bundlePreviewContainer.Visible = false;
		_bundlePreviewContainer.MouseFilter = MouseFilterEnum.Ignore;
		_cardTween?.Kill();
		_selectedBundle?.ReAddCardNodes();
		_selectedBundle?.Hitbox.TryGrabFocus();
		_previewCancelButton.Disable();
		_previewConfirmButton.Disable();
		_selectedBundle = null;
		_bundleRow.Visible = true;
	}

	private void ConfirmSelection(NButton _)
	{
		IReadOnlyList<NCard> cardNodes = _selectedBundle.CardNodes;
		foreach (NCard item in cardNodes)
		{
			NRun.Instance.GlobalUi.ReparentCard(item);
			Vector2 targetPosition = PileType.Deck.GetTargetPosition(item);
			NCardFlyVfx child = NCardFlyVfx.Create(item, targetPosition, isAddingToPile: true, item.Model.Owner.Character.TrailPath);
			NRun.Instance.GlobalUi.TopBar.TrailContainer.AddChildSafely(child);
		}
		_completionSource.SetResult(new global::_003C_003Ez__ReadOnlySingleElementList<IReadOnlyList<CardModel>>(_selectedBundle.Bundle));
	}

	public async Task<IEnumerable<IReadOnlyList<CardModel>>> CardsSelected()
	{
		IEnumerable<IReadOnlyList<CardModel>> result = await _completionSource.Task;
		NOverlayStack.Instance.Remove(this);
		return result;
	}

	public void AfterOverlayOpened()
	{
		base.Modulate = Colors.Transparent;
		_fadeTween?.Kill();
		_fadeTween = CreateTween();
		_fadeTween.TweenProperty(this, "modulate:a", 1f, 0.4);
	}

	public void AfterOverlayClosed()
	{
		_fadeTween?.Kill();
		this.QueueFreeSafely();
	}

	public void AfterOverlayShown()
	{
		base.Visible = true;
		if (_bundlePreviewContainer.Visible)
		{
			_previewCancelButton.Enable();
			_previewConfirmButton.Enable();
		}
	}

	public void AfterOverlayHidden()
	{
		base.Visible = false;
		_previewCancelButton.Disable();
		_previewConfirmButton.Disable();
	}
}
