using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Vfx;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Shops;

public partial class NMerchantCard : NMerchantSlot
{
	private Node2D _saleVisual;

	private Control _cardHolder;

	private NCard? _cardNode;

	private Tween? _hoverTween;

	private MerchantCardEntry _cardEntry;

	public bool IsShowingUpgradedCard => _cardNode?.Model?.IsUpgraded == true;

	public override MerchantEntry Entry => _cardEntry;

	protected override CanvasItem Visual => _cardHolder;

	public override void _Ready()
	{
		ConnectSignals();
		_cardHolder = GetNode<Control>("%CardHolder");
		_saleVisual = GetNode<Node2D>("%SaleVisual");
	}

	public void FillSlot(MerchantCardEntry cardEntry)
	{
		_cardEntry = cardEntry;
		cardEntry.EntryUpdated += UpdateVisual;
		cardEntry.PurchaseFailed += base.OnPurchaseFailed;
		cardEntry.PurchaseCompleted += OnSuccessfulPurchase;
		UpdateVisual();
	}

	protected override void UpdateVisual()
	{
		base.UpdateVisual();
		if (_cardEntry.CreationResult == null)
		{
			base.Visible = false;
			base.MouseFilter = MouseFilterEnum.Ignore;
			ClearHoverTip();
			return;
		}
		if (_cardNode != null && _cardNode.Model != _cardEntry.CreationResult.Card)
		{
			_cardNode.QueueFreeSafely();
			_cardNode = null;
		}
		if (_cardNode == null)
		{
			_cardNode = NCard.Create(_cardEntry.CreationResult.Card);
			_cardHolder.AddChildSafely(_cardNode);
			_cardNode.UpdateVisuals(PileType.None, CardPreviewMode.Normal);
		}
		_costLabel.SetTextAutoSize(_cardEntry.Cost.ToString());
		_saleVisual.Visible = _cardEntry.IsOnSale;
		if (!_cardEntry.EnoughGold)
		{
			_costLabel.Modulate = StsColors.red;
		}
		else
		{
			_costLabel.Modulate = (_cardEntry.IsOnSale ? StsColors.green : StsColors.cream);
		}
	}

	public void OnInventoryOpened()
	{
		CardCreationResult? creationResult = _cardEntry.CreationResult;
		if (creationResult != null && creationResult.HasBeenModified)
		{
			TaskHelper.RunSafely(DoRelicFlash());
		}
	}

	private async Task DoRelicFlash()
	{
		SceneTreeTimer sceneTreeTimer = GetTree().CreateTimer(0.4);
		await sceneTreeTimer.ToSignal(sceneTreeTimer, SceneTreeTimer.SignalName.Timeout);
		foreach (RelicModel modifyingRelic in _cardEntry.CreationResult.ModifyingRelics)
		{
			modifyingRelic.Flash();
			_cardNode?.FlashRelicOnCard(modifyingRelic);
		}
	}

	protected override async Task OnTryPurchase(MerchantInventory? inventory)
	{
		await _cardEntry.OnTryPurchaseWrapper(inventory);
	}

	protected void OnSuccessfulPurchase(PurchaseStatus _, MerchantEntry __)
	{
		TriggerMerchantHandToPointHere();
		NRun.Instance?.GlobalUi.ReparentCard(_cardNode);
		NRun.Instance?.GlobalUi.TopBar.TrailContainer.AddChildSafely(NCardFlyVfx.Create(_cardNode, PileType.Deck.GetTargetPosition(_cardNode), isAddingToPile: true, _cardNode.Model.Owner.Character.TrailPath));
		_cardNode = null;
		UpdateVisual();
	}

	protected override void OnPreview()
	{
		ClearHoverTip();
		NInspectCardScreen inspectCardScreen = NGame.Instance.GetInspectCardScreen();
		int num = 1;
		List<CardModel> list = new List<CardModel>(num);
		CollectionsMarshal.SetCount(list, num);
		Span<CardModel> span = CollectionsMarshal.AsSpan(list);
		int index = 0;
		span[index] = _cardNode.Model;
		inspectCardScreen.Open(list, 0);
	}

	protected override void CreateHoverTip()
	{
		NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, _cardEntry.CreationResult.Card.HoverTips);
		nHoverTipSet.SetAlignment(_hitbox, HoverTip.GetHoverTipAlignment(this));
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		_cardNode?.QueueFreeSafely();
		_cardEntry.EntryUpdated -= UpdateVisual;
		_cardEntry.PurchaseFailed -= base.OnPurchaseFailed;
		_cardEntry.PurchaseCompleted -= OnSuccessfulPurchase;
	}
}
