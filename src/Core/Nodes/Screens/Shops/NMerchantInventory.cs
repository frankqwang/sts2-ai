using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using Godot.Collections;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Shops;

public partial class NMerchantInventory : Control, IScreenContext
{
	[Signal]
	public delegate void InventoryClosedEventHandler();

	private const float _openPosition = 80f;

	private const float _closedPosition = -1000f;

	private Control? _characterCardContainer;

	private Control? _colorlessCardContainer;

	protected Control? _relicContainer;

	private Control? _potionContainer;

	private NMerchantCardRemoval? _cardRemovalNode;

	private NBackButton _backButton;

	private NMerchantDialogue _merchantDialogue;

	private Tween? _inventoryTween;

	private Control _slotsContainer;

	private ColorRect _backstop;

	private NMerchantSlot? _lastSlot;

	public MerchantInventory? Inventory { get; private set; }

	public bool IsOpen { get; private set; }

	public NMerchantHand MerchantHand { get; private set; }

	public Control? DefaultFocusedControl
	{
		get
		{
			NMerchantSlot lastSlot = _lastSlot;
			if (lastSlot != null)
			{
				MerchantEntry entry = lastSlot.Entry;
				if (entry != null && entry.IsStocked)
				{
					return _lastSlot;
				}
			}
			return GetAllSlots().FirstOrDefault((NMerchantSlot s) => s.Entry.IsStocked);
		}
	}

	public override void _Ready()
	{
		_merchantDialogue = GetNode<NMerchantDialogue>("%Dialogue");
		_merchantDialogue.Modulate = Colors.Transparent;
		_slotsContainer = GetNode<Control>("%SlotsContainer");
		_slotsContainer.Position = new Vector2(_slotsContainer.Position.X, -1000f);
		_backstop = GetNode<ColorRect>("Backstop");
		MerchantHand = GetNode<NMerchantHand>("%MerchantHand");
		_characterCardContainer = GetNodeOrNull<Control>("%CharacterCards");
		_colorlessCardContainer = GetNodeOrNull<Control>("%ColorlessCards");
		_relicContainer = GetNodeOrNull<Control>("%Relics");
		_potionContainer = GetNodeOrNull<Control>("%Potions");
		_cardRemovalNode = GetNodeOrNull<NMerchantCardRemoval>("%MerchantCardRemoval");
		_backButton = GetNode<NBackButton>("%BackButton");
		_backButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(delegate
		{
			Close();
		}));
		_backButton.Disable();
		NGame.Instance.SetScreenShakeTarget(this);
	}

	public override void _EnterTree()
	{
		ActiveScreenContext.Instance.Updated += OnActiveScreenUpdated;
		SubscribeToEntries();
	}

	public override void _ExitTree()
	{
		ActiveScreenContext.Instance.Updated -= OnActiveScreenUpdated;
		if (Inventory == null)
		{
			return;
		}
		foreach (MerchantEntry allEntry in Inventory.AllEntries)
		{
			allEntry.PurchaseCompleted -= OnPurchaseCompleted;
			allEntry.PurchaseFailed -= _merchantDialogue.ShowForPurchaseAttempt;
		}
	}

	public void Initialize(MerchantInventory inventory, MerchantDialogueSet dialogue)
	{
		if (Inventory != null)
		{
			throw new InvalidOperationException("Merchant inventory already populated.");
		}
		Inventory = inventory;
		for (int i = 0; i < Inventory.CharacterCardEntries.Count; i++)
		{
			NMerchantCard child = _characterCardContainer.GetChild<NMerchantCard>(i);
			child.Initialize(this);
			child.FillSlot(Inventory.CharacterCardEntries[i]);
		}
		for (int j = 0; j < Inventory.ColorlessCardEntries.Count; j++)
		{
			NMerchantCard child2 = _colorlessCardContainer.GetChild<NMerchantCard>(j);
			child2.Initialize(this);
			child2.FillSlot(Inventory.ColorlessCardEntries[j]);
		}
		for (int k = 0; k < Inventory.RelicEntries.Count; k++)
		{
			NMerchantRelic child3 = _relicContainer.GetChild<NMerchantRelic>(k);
			child3.Initialize(this);
			child3.FillSlot(Inventory.RelicEntries[k]);
		}
		for (int l = 0; l < Inventory.PotionEntries.Count; l++)
		{
			NMerchantPotion child4 = _potionContainer.GetChild<NMerchantPotion>(l);
			child4.Initialize(this);
			child4.FillSlot(Inventory.PotionEntries[l]);
		}
		if (Inventory.CardRemovalEntry != null)
		{
			_cardRemovalNode.Initialize(this);
			_cardRemovalNode.FillSlot(Inventory.CardRemovalEntry);
		}
		SubscribeToEntries();
		UpdateNavigation();
		foreach (NMerchantSlot slot in GetAllSlots())
		{
			slot.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
			{
				_lastSlot = slot;
			}));
		}
		_merchantDialogue.Initialize(dialogue);
	}

	public void Open()
	{
		if (!SaveManager.Instance.SeenFtue("merchant_ftue"))
		{
			SaveManager.Instance.MarkFtueAsComplete("merchant_ftue");
		}
		TaskHelper.RunSafely(DoOpenAnimation());
		base.MouseFilter = MouseFilterEnum.Stop;
		_backButton.Enable();
		foreach (NMerchantCard cardSlot in GetCardSlots())
		{
			cardSlot.OnInventoryOpened();
		}
		SfxCmd.Play("event:/sfx/npcs/merchant/merchant_welcome");
		IsOpen = true;
		ActiveScreenContext.Instance.Update();
		_merchantDialogue.ShowOnInventoryOpen();
	}

	private void SubscribeToEntries()
	{
		if (!IsNodeReady() || Inventory == null)
		{
			return;
		}
		foreach (MerchantEntry allEntry in Inventory.AllEntries)
		{
			allEntry.PurchaseCompleted += OnPurchaseCompleted;
			allEntry.PurchaseFailed += _merchantDialogue.ShowForPurchaseAttempt;
		}
	}

	private async Task DoOpenAnimation()
	{
		_inventoryTween?.Kill();
		_inventoryTween = CreateTween().SetParallel();
		_inventoryTween.TweenProperty(_backstop, "modulate:a", 0.8f, 1.0).SetEase(Tween.EaseType.InOut).SetTrans(Tween.TransitionType.Sine)
			.FromCurrent();
		_inventoryTween.TweenProperty(_slotsContainer, "position:y", 80f, 0.699999988079071).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Quint)
			.FromCurrent();
		await ToSignal(_inventoryTween, Tween.SignalName.Finished);
	}

	private void Close()
	{
		MerchantHand.StopPointing(0f);
		_inventoryTween?.Kill();
		_inventoryTween = CreateTween().SetParallel();
		_inventoryTween.TweenProperty(_backstop, "modulate:a", 0f, 0.800000011920929).SetEase(Tween.EaseType.InOut).SetTrans(Tween.TransitionType.Sine)
			.FromCurrent();
		_inventoryTween.TweenProperty(_slotsContainer, "position:y", -1000f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic)
			.FromCurrent();
		base.MouseFilter = MouseFilterEnum.Ignore;
		_backButton.Disable();
		_lastSlot = null;
		IsOpen = false;
		ActiveScreenContext.Instance.Update();
		EmitSignalInventoryClosed();
	}

	private void OnPurchaseCompleted(PurchaseStatus status, MerchantEntry entry)
	{
		UpdateNavigation();
		NMerchantSlot lastSlot = GetAllSlots().FirstOrDefault((NMerchantSlot s) => s.Entry == entry);
		if (lastSlot != null)
		{
			(from s in GetAllSlots()
				where s.Visible && s != lastSlot
				select s).MinBy((NMerchantSlot s) => (s.GlobalPosition - lastSlot.GlobalPosition).Length())?.TryGrabFocus();
		}
		SfxCmd.Play("event:/sfx/npcs/merchant/merchant_thank_yous");
		_merchantDialogue.ShowForPurchaseAttempt(status);
	}

	public void OnCardRemovalUsed()
	{
		_cardRemovalNode.OnCardRemovalUsed();
	}

	public IEnumerable<NMerchantSlot> GetAllSlots()
	{
		List<NMerchantSlot> list = new List<NMerchantSlot>();
		list.AddRange(GetCardSlots());
		if (_relicContainer != null)
		{
			list.AddRange(_relicContainer.GetChildren().OfType<NMerchantRelic>());
		}
		if (_potionContainer != null)
		{
			list.AddRange(_potionContainer.GetChildren().OfType<NMerchantPotion>());
		}
		if (_cardRemovalNode != null)
		{
			list.Add(_cardRemovalNode);
		}
		return list;
	}

	private IEnumerable<NMerchantCard> GetCardSlots()
	{
		return new IEnumerable<Node>[2]
		{
			_characterCardContainer?.GetChildren() ?? new Array<Node>(),
			_colorlessCardContainer?.GetChildren() ?? new Array<Node>()
		}.SelectMany((IEnumerable<Node> n) => n).OfType<NMerchantCard>();
	}

	protected virtual void UpdateNavigation()
	{
		UpdateHorizontalNavigation();
		UpdateVerticalNavigation();
	}

	private void UpdateHorizontalNavigation()
	{
		List<NMerchantSlot> source = (from c in _characterCardContainer?.GetChildren().OfType<NMerchantSlot>()
			where c.Visible
			select c).ToList() ?? new List<NMerchantSlot>();
		List<NMerchantSlot> list = (from c in _colorlessCardContainer?.GetChildren().OfType<NMerchantSlot>()
			where c.Visible
			select c).ToList() ?? new List<NMerchantSlot>();
		List<NMerchantSlot> list2 = (from c in _relicContainer?.GetChildren().OfType<NMerchantSlot>()
			where c.Visible
			select c).ToList() ?? new List<NMerchantSlot>();
		List<NMerchantSlot> list3 = (from c in _potionContainer?.GetChildren().OfType<NMerchantSlot>()
			where c.Visible
			select c).ToList() ?? new List<NMerchantSlot>();
		List<NMerchantSlot> list4 = source.ToList();
		IEnumerable<NMerchantSlot> first = list.Concat(list2);
		IEnumerable<NMerchantSlot> second;
		if (_cardRemovalNode == null)
		{
			IEnumerable<NMerchantSlot> enumerable = System.Array.Empty<NMerchantSlot>();
			second = enumerable;
		}
		else
		{
			IEnumerable<NMerchantSlot> enumerable = new global::_003C_003Ez__ReadOnlySingleElementList<NMerchantSlot>(_cardRemovalNode);
			second = enumerable;
		}
		List<NMerchantSlot> list5 = first.Concat(second).ToList();
		IEnumerable<NMerchantSlot> first2 = list.Concat(list3);
		IEnumerable<NMerchantSlot> second2;
		if (_cardRemovalNode == null)
		{
			IEnumerable<NMerchantSlot> enumerable = System.Array.Empty<NMerchantSlot>();
			second2 = enumerable;
		}
		else
		{
			IEnumerable<NMerchantSlot> enumerable = new global::_003C_003Ez__ReadOnlySingleElementList<NMerchantSlot>(_cardRemovalNode);
			second2 = enumerable;
		}
		List<NMerchantSlot> list6 = first2.Concat(second2).ToList();
		for (int num = 0; num < list4.Count; num++)
		{
			list4[num].FocusNeighborLeft = ((num > 0) ? list4[num - 1].GetPath() : list4[num].GetPath());
			list4[num].FocusNeighborRight = ((num < list4.Count - 1) ? list4[num + 1].GetPath() : list4[num].GetPath());
		}
		for (int num2 = 0; num2 < list6.Count; num2++)
		{
			list6[num2].FocusNeighborLeft = ((num2 > 0) ? list6[num2 - 1].GetPath() : list6[num2].GetPath());
			list6[num2].FocusNeighborRight = ((num2 < list6.Count - 1) ? list6[num2 + 1].GetPath() : list6[num2].GetPath());
		}
		for (int num3 = 0; num3 < list5.Count; num3++)
		{
			list5[num3].FocusNeighborLeft = ((num3 > 0) ? list5[num3 - 1].GetPath() : list5[num3].GetPath());
			list5[num3].FocusNeighborRight = ((num3 < list5.Count - 1) ? list5[num3 + 1].GetPath() : list5[num3].GetPath());
		}
		if (list2.Count == 0 && list3.Count > 0)
		{
			if (_cardRemovalNode != null)
			{
				_cardRemovalNode.FocusNeighborLeft = list3.Last().GetPath();
			}
			if (list.Count > 0)
			{
				list.Last().FocusNeighborRight = list3.First().GetPath();
			}
		}
	}

	private void UpdateVerticalNavigation()
	{
		List<NMerchantSlot> source = _characterCardContainer?.GetChildren().OfType<NMerchantSlot>().ToList() ?? new List<NMerchantSlot>();
		List<NMerchantSlot> first = _colorlessCardContainer?.GetChildren().OfType<NMerchantSlot>().ToList() ?? new List<NMerchantSlot>();
		List<NMerchantSlot> second = _relicContainer?.GetChildren().OfType<NMerchantSlot>().ToList() ?? new List<NMerchantSlot>();
		List<NMerchantSlot> second2 = _potionContainer?.GetChildren().OfType<NMerchantSlot>().ToList() ?? new List<NMerchantSlot>();
		List<NMerchantSlot> list = source.ToList();
		IEnumerable<NMerchantSlot> first2 = first.Concat(second);
		IEnumerable<NMerchantSlot> second3;
		if (_cardRemovalNode == null)
		{
			IEnumerable<NMerchantSlot> enumerable = System.Array.Empty<NMerchantSlot>();
			second3 = enumerable;
		}
		else
		{
			IEnumerable<NMerchantSlot> enumerable = new global::_003C_003Ez__ReadOnlySingleElementList<NMerchantSlot>(_cardRemovalNode);
			second3 = enumerable;
		}
		List<NMerchantSlot> list2 = first2.Concat(second3).ToList();
		IEnumerable<NMerchantSlot> first3 = first.Concat(second2);
		IEnumerable<NMerchantSlot> second4;
		if (_cardRemovalNode == null)
		{
			IEnumerable<NMerchantSlot> enumerable = System.Array.Empty<NMerchantSlot>();
			second4 = enumerable;
		}
		else
		{
			IEnumerable<NMerchantSlot> enumerable = new global::_003C_003Ez__ReadOnlySingleElementList<NMerchantSlot>(_cardRemovalNode);
			second4 = enumerable;
		}
		List<NMerchantSlot> list3 = first3.Concat(second4).ToList();
		for (int i = 0; i < list.Count; i++)
		{
			list[i].FocusNeighborTop = list[i].GetPath();
			if (list2.Count > 0)
			{
				Control closestVisible = GetClosestVisible(i, list2);
				if (closestVisible != null)
				{
					list[i].FocusNeighborBottom = closestVisible.GetPath();
					continue;
				}
			}
			Control closestVisible2 = GetClosestVisible(i, list3);
			if (closestVisible2 != null)
			{
				list[i].FocusNeighborBottom = closestVisible2.GetPath();
			}
			else
			{
				list[i].FocusNeighborBottom = list[i].GetPath();
			}
		}
		for (int j = 0; j < list3.Count; list3[j].FocusNeighborBottom = list3[j].GetPath(), j++)
		{
			if (list2.Count > 0)
			{
				Control closestVisible3 = GetClosestVisible(j, list2);
				if (closestVisible3 != null)
				{
					list3[j].FocusNeighborTop = closestVisible3.GetPath();
					continue;
				}
			}
			Control closestVisible4 = GetClosestVisible(j, list);
			if (closestVisible4 != null)
			{
				list3[j].FocusNeighborTop = closestVisible4.GetPath();
			}
			else
			{
				list3[j].FocusNeighborTop = list3[j].GetPath();
			}
		}
		for (int k = 0; k < list2.Count; k++)
		{
			if (list.Count > 0)
			{
				Control closestVisible5 = GetClosestVisible(k, list);
				if (closestVisible5 != null)
				{
					list2[k].FocusNeighborTop = closestVisible5.GetPath();
					goto IL_02bb;
				}
			}
			list2[k].FocusNeighborTop = list2[k].GetPath();
			goto IL_02bb;
			IL_02bb:
			if (list3.Count > 0)
			{
				Control closestVisible6 = GetClosestVisible(k, list3);
				if (closestVisible6 != null)
				{
					list2[k].FocusNeighborBottom = closestVisible6.GetPath();
					continue;
				}
			}
			list2[k].FocusNeighborBottom = list2[k].GetPath();
		}
	}

	private Control? GetClosestVisible(int idx, List<NMerchantSlot> row)
	{
		NMerchantSlot nMerchantSlot = row[Math.Min(idx, row.Count - 1)];
		if (nMerchantSlot.Visible)
		{
			return nMerchantSlot;
		}
		int num = row.IndexOf(nMerchantSlot);
		int num2 = num - 1;
		int num3 = num + 1;
		while (num2 >= 0 || num3 < row.Count)
		{
			if (num3 < row.Count)
			{
				if (row[num3].Visible)
				{
					return row[num3];
				}
				num3++;
			}
			if (num2 >= 0)
			{
				if (row[num2].Visible)
				{
					return row[num2];
				}
				num2--;
			}
		}
		return null;
	}

	private void OnActiveScreenUpdated()
	{
		this.UpdateControllerNavEnabled();
		if (ActiveScreenContext.Instance.IsCurrent(this))
		{
			if (_characterCardContainer != null && NControllerManager.Instance.IsUsingController && _inventoryTween != null && _inventoryTween.IsRunning())
			{
				float num = 80f - _slotsContainer.Position.Y;
				MerchantHand.PointAtTarget(_characterCardContainer.GetChild<NMerchantCard>(0).GlobalPosition + Vector2.Down * num);
			}
			_backButton.Enable();
		}
		else
		{
			_backButton.Disable();
		}
	}
}
