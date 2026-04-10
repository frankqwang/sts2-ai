using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Extensions;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Ftue;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Potions;

public partial class NPotionContainer : Control
{
	private Player? _player;

	private readonly List<NPotionHolder> _holders = new List<NPotionHolder>();

	private Control _potionHolders;

	private Control _potionErrorBg;

	private NButton _potionShortcutButton;

	private Tween? _potionsFullTween;

	private Vector2 _potionHolderInitPos;

	private NPotionHolder? _focusedHolder;

	public Control? FirstPotionControl => _holders.FirstOrDefault();

	public Control? LastPotionControl => _holders.LastOrDefault();

	public override void _Ready()
	{
		Callable.From(UpdateNavigation).CallDeferred();
	}

	public override void _EnterTree()
	{
		_potionHolders = GetNode<Control>("MarginContainer/PotionHolders");
		_potionErrorBg = GetNode<Control>("PotionErrorBg");
		_potionShortcutButton = GetNode<NButton>("PotionShortcutButton");
		_potionErrorBg.Modulate = Colors.Transparent;
		_potionShortcutButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(delegate
		{
			_potionHolders.GetChild<Control>(0).TryGrabFocus();
		}));
		CombatManager.Instance.CombatSetUp += OnCombatSetUp;
		ConnectPlayerEvents();
	}

	public override void _ExitTree()
	{
		DisconnectPlayerEvents();
		_player = null;
		CombatManager.Instance.CombatSetUp -= OnCombatSetUp;
	}

	public void Initialize(IRunState runState)
	{
		DisconnectPlayerEvents();
		_player = LocalContext.GetMe(runState);
		ConnectPlayerEvents();
		GrowPotionHolders(_player.MaxPotionCount);
		foreach (PotionModel potion in _player.Potions)
		{
			Add(potion, isInitialization: true);
		}
	}

	private void ConnectPlayerEvents()
	{
		if (_player != null)
		{
			_player.AddPotionFailed += PlayAddFailedAnim;
			_player.PotionProcured += OnPotionProcured;
			_player.UsedPotionRemoved += OnUsedPotionRemoved;
			_player.PotionDiscarded += Discard;
			_player.MaxPotionCountChanged += GrowPotionHolders;
			_player.RelicObtained += OnRelicsUpdated;
			_player.RelicRemoved += OnRelicsUpdated;
		}
	}

	private void DisconnectPlayerEvents()
	{
		if (_player != null)
		{
			_player.AddPotionFailed -= PlayAddFailedAnim;
			_player.PotionProcured -= OnPotionProcured;
			_player.UsedPotionRemoved -= OnUsedPotionRemoved;
			_player.PotionDiscarded -= Discard;
			_player.MaxPotionCountChanged -= GrowPotionHolders;
			_player.RelicObtained -= OnRelicsUpdated;
			_player.RelicRemoved -= OnRelicsUpdated;
		}
	}

	private void GrowPotionHolders(int newMaxPotionSlots)
	{
		for (int i = _holders.Count; i < newMaxPotionSlots; i++)
		{
			NPotionHolder node = NPotionHolder.Create(isUsable: true);
			_holders.Add(node);
			_potionHolders.AddChildSafely(node);
			node.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
			{
				OnPotionHolderFocused(node);
			}));
			node.Connect(Control.SignalName.FocusExited, Callable.From(delegate
			{
				OnPotionHolderUnfocused(node);
			}));
			node.Connect(Control.SignalName.MouseEntered, Callable.From(delegate
			{
				OnPotionHolderFocused(node);
			}));
			node.Connect(Control.SignalName.MouseExited, Callable.From(delegate
			{
				OnPotionHolderUnfocused(node);
			}));
		}
		UpdateNavigation();
	}

	private void OnRelicsUpdated(RelicModel _)
	{
		Callable.From(UpdateNavigation).CallDeferred();
	}

	private void UpdateNavigation()
	{
		Control control = NRun.Instance.GlobalUi.RelicInventory.RelicNodes.FirstOrDefault();
		if (control != null)
		{
			for (int i = 0; i < _holders.Count; i++)
			{
				_holders[i].FocusNeighborLeft = ((i > 0) ? _holders[i - 1].GetPath() : NRun.Instance.GlobalUi.TopBar.Gold.GetPath());
				_holders[i].FocusNeighborRight = ((i < _holders.Count - 1) ? _holders[i + 1].GetPath() : NRun.Instance.GlobalUi.TopBar.RoomIcon.GetPath());
				_holders[i].FocusNeighborBottom = control.GetPath();
				_holders[i].FocusNeighborTop = _holders[i].GetPath();
			}
		}
	}

	private void Add(PotionModel potion, bool isInitialization)
	{
		if (!_holders.All((NPotionHolder h) => h.HasPotion))
		{
			if (!isInitialization)
			{
				PotionFtueCheck();
			}
			NPotion nPotion = NPotion.Create(potion);
			nPotion.Position = new Vector2(-30f, -30f);
			NPotionHolder nPotionHolder = _holders[potion.Owner.PotionSlots.IndexOf<PotionModel>(potion)];
			nPotionHolder.AddPotion(nPotion);
		}
	}

	public void AnimatePotion(PotionModel potion, Vector2? startPosition = null)
	{
		if (LocalContext.IsMine(potion))
		{
			NPotionHolder nPotionHolder = _holders.First((NPotionHolder n) => n.Potion != null && n.Potion.Model == potion);
			TaskHelper.RunSafely(nPotionHolder.Potion.PlayNewlyAcquiredAnimation(startPosition));
		}
	}

	public void OnPotionUseCanceled(PotionModel potion)
	{
		NPotionHolder nPotionHolder = _holders.FirstOrDefault((NPotionHolder n) => n.Potion?.Model == potion);
		if (nPotionHolder != null)
		{
			nPotionHolder.CancelPotionUse();
			return;
		}
		Log.Error($"Tried to cancel potion use for potion {potion} but a holder for it does not exist in the player's belt!");
	}

	private void PotionFtueCheck()
	{
		if (!SaveManager.Instance.SeenFtue("obtain_potion_ftue"))
		{
			NModalContainer.Instance.Add(NObtainPotionFtue.Create());
			SaveManager.Instance.MarkFtueAsComplete("obtain_potion_ftue");
		}
	}

	private void PlayAddFailedAnim()
	{
		if (_potionsFullTween != null && _potionsFullTween.IsRunning())
		{
			_potionsFullTween?.Kill();
			_potionHolders.Position = _potionHolderInitPos;
		}
		_potionsFullTween = CreateTween().SetParallel();
		_potionHolderInitPos = _potionHolders.Position;
		_potionsFullTween.TweenMethod(Callable.From(delegate(float t)
		{
			_potionHolders.Position = _potionHolderInitPos + Vector2.Right * 3f * Mathf.Sin(t * 5f) * Mathf.Sin(t * 0.5f);
		}), 0f, (float)Math.PI * 2f, 0.5);
		_potionsFullTween.TweenProperty(_potionErrorBg, "modulate", Colors.White, 0.15);
		_potionsFullTween.TweenProperty(_potionErrorBg, "modulate", Colors.Transparent, 0.5).SetDelay(0.35);
	}

	private void Discard(PotionModel potion)
	{
		NPotionHolder nPotionHolder = _holders.First((NPotionHolder n) => n.Potion != null && n.Potion.Model == potion);
		OnPotionHolderUnfocused(nPotionHolder);
		nPotionHolder.DiscardPotion();
	}

	private void RemoveUsed(PotionModel potion)
	{
		NPotionHolder nPotionHolder = _holders.First((NPotionHolder n) => n.Potion != null && n.Potion.Model == potion);
		OnPotionHolderUnfocused(nPotionHolder);
		nPotionHolder.RemoveUsedPotion();
	}

	private void OnPotionProcured(PotionModel potion)
	{
		Add(potion, isInitialization: false);
	}

	private void OnUsedPotionRemoved(PotionModel potion)
	{
		RemoveUsed(potion);
	}

	private void OnPotionHolderFocused(NPotionHolder holder)
	{
		if (_focusedHolder != holder && holder.Potion != null)
		{
			RunManager.Instance.HoveredModelTracker.OnLocalPotionHovered(holder.Potion.Model);
			_focusedHolder = holder;
		}
	}

	private void OnPotionHolderUnfocused(NPotionHolder holder)
	{
		if (_focusedHolder == holder)
		{
			RunManager.Instance.HoveredModelTracker.OnLocalPotionUnhovered();
			_focusedHolder = null;
		}
	}

	private void OnCombatSetUp(CombatState _)
	{
		TaskHelper.RunSafely(ShinePotions());
	}

	private async Task ShinePotions()
	{
		await Cmd.Wait(1f);
		foreach (NPotionHolder holder in _holders)
		{
			await TaskHelper.RunSafely(holder.ShineOnStartOfCombat());
		}
	}
}
