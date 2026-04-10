using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.HoverTips;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Shops;

public partial class NMerchantCardRemoval : NMerchantSlot
{
	private const string _locTable = "merchant_room";

	private Sprite2D _removalVisual;

	private AnimationPlayer _animator;

	private Control _costContainer;

	private bool _isUnavailable;

	private MerchantCardRemovalEntry _removalEntry;

	private LocString Title => new LocString("merchant_room", "MERCHANT.cardRemovalService.title");

	private LocString Description => new LocString("merchant_room", "MERCHANT.cardRemovalService.description");

	public override MerchantEntry Entry => _removalEntry;

	protected override CanvasItem Visual => _removalVisual;

	public override void _Ready()
	{
		ConnectSignals();
		_removalVisual = GetNode<Sprite2D>("%Visual");
		_animator = GetNode<AnimationPlayer>("%Animation");
		_costContainer = GetNode<Control>("Cost");
	}

	public void FillSlot(MerchantCardRemovalEntry removalEntry)
	{
		_removalEntry = removalEntry;
		_removalEntry.EntryUpdated += UpdateVisual;
		_removalEntry.PurchaseFailed += base.OnPurchaseFailed;
		_removalEntry.PurchaseCompleted += OnSuccessfulPurchase;
		if (!Hook.ShouldAllowMerchantCardRemoval(base.Player.RunState, base.Player))
		{
			_removalEntry.SetUsed();
		}
		UpdateVisual();
	}

	protected override void UpdateVisual()
	{
		base.UpdateVisual();
		if (!_isUnavailable)
		{
			if (_removalEntry.Used)
			{
				_hitbox.MouseFilter = MouseFilterEnum.Ignore;
				_animator.CurrentAnimation = "Used";
				_isUnavailable = true;
				_animator.Play();
				_costLabel.Visible = false;
				_costContainer.Visible = false;
				base.FocusMode = FocusModeEnum.None;
			}
			else
			{
				base.MouseFilter = MouseFilterEnum.Stop;
				_costLabel.Visible = true;
				_costLabel.SetTextAutoSize(_removalEntry.Cost.ToString());
				_costLabel.Modulate = (_removalEntry.EnoughGold ? StsColors.cream : StsColors.red);
				_costContainer.Visible = true;
				base.FocusMode = FocusModeEnum.All;
			}
			ClearHoverTip();
		}
	}

	protected override async Task OnTryPurchase(MerchantInventory? inventory)
	{
		await _removalEntry.OnTryPurchaseWrapper(inventory);
	}

	protected void OnSuccessfulPurchase(PurchaseStatus _, MerchantEntry __)
	{
		TriggerMerchantHandToPointHere();
		UpdateVisual();
	}

	public void OnCardRemovalUsed()
	{
		_removalEntry.SetUsed();
		UpdateVisual();
	}

	protected override void CreateHoverTip()
	{
		LocString title = Title;
		LocString description = Description;
		description.Add("Amount", _removalEntry.CalcPriceIncrease());
		NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, new HoverTip(title, description));
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
		_removalEntry.EntryUpdated -= UpdateVisual;
		_removalEntry.PurchaseFailed -= base.OnPurchaseFailed;
		_removalEntry.PurchaseCompleted -= OnSuccessfulPurchase;
	}
}
