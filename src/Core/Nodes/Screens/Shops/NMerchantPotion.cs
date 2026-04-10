using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Potions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Shops;

public partial class NMerchantPotion : NMerchantSlot
{
	private Control _potionHolder;

	private NPotion? _potionNode;

	private MerchantPotionEntry _potionEntry;

	private PotionModel? _potion;

	private Vector2 _potionNodePosition;

	public override MerchantEntry Entry => _potionEntry;

	protected override CanvasItem Visual => _potionHolder;

	public override void _Ready()
	{
		ConnectSignals();
		_potionHolder = GetNode<Control>("%PotionHolder");
	}

	public void FillSlot(MerchantPotionEntry potionEntry)
	{
		_potionEntry = potionEntry;
		_potion = potionEntry.Model;
		_potionEntry.EntryUpdated += UpdateVisual;
		_potionEntry.PurchaseFailed += base.OnPurchaseFailed;
		_potionEntry.PurchaseCompleted += OnSuccessfulPurchase;
		UpdateVisual();
	}

	protected override void UpdateVisual()
	{
		base.UpdateVisual();
		if (_potionEntry.Model == null)
		{
			base.Visible = false;
			base.MouseFilter = MouseFilterEnum.Ignore;
			if (_potionNode != null)
			{
				_potionNode.QueueFreeSafely();
				_potionNode = null;
			}
			ClearHoverTip();
			return;
		}
		if (_potionNode != null && _potionNode.Model != _potionEntry.Model)
		{
			_potionNode.QueueFreeSafely();
			_potionNode = null;
		}
		if (_potionNode == null)
		{
			_potionNode = NPotion.Create(_potionEntry.Model);
			_potionHolder.AddChildSafely(_potionNode);
			_potionNode.Position = Vector2.Zero;
		}
		_potionNodePosition = _potionNode.GlobalPosition;
		_costLabel.SetTextAutoSize(_potionEntry.Cost.ToString());
		_costLabel.Modulate = (_potionEntry.EnoughGold ? StsColors.cream : StsColors.red);
	}

	protected override async Task OnTryPurchase(MerchantInventory? inventory)
	{
		await _potionEntry.OnTryPurchaseWrapper(inventory);
	}

	protected void OnSuccessfulPurchase(PurchaseStatus _, MerchantEntry __)
	{
		TriggerMerchantHandToPointHere();
		NRun.Instance?.GlobalUi.TopBar.PotionContainer.AnimatePotion(_potion, _potionNodePosition);
		UpdateVisual();
		_potion = _potionEntry.Model;
	}

	protected override void CreateHoverTip()
	{
		NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, _potionNode.Model.HoverTips);
		nHoverTipSet.GlobalPosition = base.GlobalPosition;
		if (base.GlobalPosition.X > GetViewport().GetVisibleRect().Size.X * 0.5f)
		{
			nHoverTipSet.SetAlignment(this, HoverTipAlignment.Left);
			nHoverTipSet.GlobalPosition -= base.Size * 0.5f * base.Scale;
		}
		else
		{
			nHoverTipSet.GlobalPosition += Vector2.Right * base.Size.X * 0.5f * base.Scale + Vector2.Up * base.Size.Y * 0.5f * base.Scale;
		}
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		_potionEntry.EntryUpdated -= UpdateVisual;
		_potionEntry.PurchaseFailed -= base.OnPurchaseFailed;
		_potionEntry.PurchaseCompleted -= OnSuccessfulPurchase;
	}
}
