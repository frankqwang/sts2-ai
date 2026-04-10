using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Relics;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Shops;

public partial class NMerchantRelic : NMerchantSlot
{
	[Export(PropertyHint.None, "")]
	private NRelic.IconSize _iconSize;

	private Control _relicHolder;

	private NRelic? _relicNode;

	private MerchantRelicEntry _relicEntry;

	private RelicModel? _relic;

	private Vector2 _relicNodePosition;

	public override MerchantEntry Entry => _relicEntry;

	protected override CanvasItem Visual => _relicHolder;

	public override void _Ready()
	{
		ConnectSignals();
		_relicHolder = GetNode<Control>("%RelicHolder");
	}

	public void FillSlot(MerchantRelicEntry relicEntry)
	{
		_relicEntry = relicEntry;
		_relic = relicEntry.Model;
		relicEntry.EntryUpdated += UpdateVisual;
		relicEntry.PurchaseFailed += base.OnPurchaseFailed;
		relicEntry.PurchaseCompleted += OnSuccessfulPurchase;
		UpdateVisual();
	}

	protected override void UpdateVisual()
	{
		base.UpdateVisual();
		if (_relicEntry.Model == null)
		{
			base.Visible = false;
			base.MouseFilter = MouseFilterEnum.Ignore;
			if (_relicNode != null)
			{
				_relicNode.QueueFreeSafely();
				_relicNode = null;
			}
			ClearHoverTip();
			return;
		}
		if (_relicNode != null && _relicNode.Model != _relicEntry.Model)
		{
			_relicNode.QueueFreeSafely();
			_relicNode = null;
		}
		if (_relicNode == null)
		{
			_relicNode = NRelic.Create(_relicEntry.Model, _iconSize);
			_relicHolder.AddChildSafely(_relicNode);
			if (_iconSize == NRelic.IconSize.Large)
			{
				_relicNode.Size = new Vector2(128f, 128f);
				_relicNode.Icon.Position -= new Vector2(0f, _costLabel.Size.Y);
			}
			base.Hitbox.Size = _relicNode.Icon.Size;
			base.Hitbox.Scale = _relicHolder.Scale;
			base.Hitbox.GlobalPosition = _relicNode.Icon.GlobalPosition;
		}
		_relicNodePosition = _relicNode.Icon.GlobalPosition;
		_costLabel.SetTextAutoSize(_relicEntry.Cost.ToString());
		_costLabel.Modulate = (_relicEntry.EnoughGold ? StsColors.cream : StsColors.red);
	}

	protected override async Task OnTryPurchase(MerchantInventory? inventory)
	{
		await _relicEntry.OnTryPurchaseWrapper(inventory);
	}

	private void OnSuccessfulPurchase(PurchaseStatus _, MerchantEntry __)
	{
		TriggerMerchantHandToPointHere();
		NRun.Instance?.GlobalUi.RelicInventory.AnimateRelic(_relic, _relicNodePosition);
		UpdateVisual();
		_relic = _relicEntry.Model;
	}

	protected override void CreateHoverTip()
	{
		NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, _relicNode.Model.HoverTips);
		nHoverTipSet.GlobalPosition = base.GlobalPosition;
		if (base.GlobalPosition.X > GetViewport().GetVisibleRect().Size.X * 0.5f)
		{
			nHoverTipSet.SetAlignment(this, HoverTipAlignment.Left);
			nHoverTipSet.GlobalPosition -= base.Size * 0.5f * base.Scale;
		}
		else
		{
			nHoverTipSet.SetAlignment(this, HoverTipAlignment.Right);
			nHoverTipSet.GlobalPosition += Vector2.Right * base.Size.X * 0.5f * base.Scale + Vector2.Up * base.Size.Y * 0.5f * base.Scale;
		}
	}

	protected override void OnPreview()
	{
		ClearHoverTip();
		int num = 1;
		List<RelicModel> list = new List<RelicModel>(num);
		CollectionsMarshal.SetCount(list, num);
		Span<RelicModel> span = CollectionsMarshal.AsSpan(list);
		int index = 0;
		span[index] = _relicNode.Model;
		List<RelicModel> relics = list;
		NGame.Instance.GetInspectRelicScreen().Open(relics, _relicNode.Model);
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		_relicEntry.EntryUpdated -= UpdateVisual;
		_relicEntry.PurchaseFailed -= base.OnPurchaseFailed;
		_relicEntry.PurchaseCompleted -= OnSuccessfulPurchase;
	}
}
