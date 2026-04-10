using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models.Potions;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Multiplayer;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.Capstones;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Potions;

public partial class NPotionHolder : NClickableControl
{
	private Vector2 _potionScale = 0.9f * Vector2.One;

	private static readonly HoverTip _emptyHoverTip = new HoverTip(new LocString("static_hover_tips", "POTION_SLOT.title"), new LocString("static_hover_tips", "POTION_SLOT.description"));

	private TextureRect _emptyIcon;

	private NPotionPopup? _popup;

	private bool _potionTargeting;

	private bool _isUsable;

	private Tween? _emptyPotionTween;

	private Tween? _hoverTween;

	private bool _disabledUntilPotionRemoved;

	private bool _isFocused;

	private CancellationTokenSource? _cancelGrayOutPotionSource;

	public NPotion? Potion { get; private set; }

	public bool HasPotion => Potion != null;

	public bool IsPotionUsable => _popup.IsUsable;

	private static string ScenePath => SceneHelper.GetScenePath("/potions/potion_holder");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	public static NPotionHolder Create(bool isUsable)
	{
		NPotionHolder nPotionHolder = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NPotionHolder>(PackedScene.GenEditState.Disabled);
		nPotionHolder._isUsable = isUsable;
		return nPotionHolder;
	}

	public override void _Ready()
	{
		_emptyIcon = GetNode<TextureRect>("%EmptyIcon");
		ConnectSignals();
	}

	protected override void OnFocus()
	{
		if (_isFocused)
		{
			return;
		}
		_isFocused = true;
		_hoverTween?.Kill();
		_hoverTween = CreateTween().SetParallel();
		if (Potion != null)
		{
			Potion.DoBounce();
			_hoverTween.TweenProperty(Potion, "scale", _potionScale * 1.15f, 0.05);
			NDebugAudioManager.Instance?.Play(Rng.Chaotic.NextItem(TmpSfx.PotionSlosh), 0.5f, PitchVariance.Large);
			if (!GodotObject.IsInstanceValid(_popup))
			{
				NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, Potion.Model.HoverTips, HoverTipAlignment.Center);
				nHoverTipSet.GlobalPosition = base.GlobalPosition + Vector2.Down * base.Size.Y * Mathf.Max(1.5f, base.Scale.Y);
			}
		}
		else
		{
			_hoverTween.TweenProperty(_emptyIcon, "scale", _potionScale * 1.15f, 0.05);
			NHoverTipSet nHoverTipSet2 = NHoverTipSet.CreateAndShow(this, _emptyHoverTip);
			nHoverTipSet2.GlobalPosition = base.GlobalPosition + Vector2.Down * base.Size.Y * Mathf.Max(1.5f, base.Scale.Y);
			nHoverTipSet2.SetAlignment(this, HoverTipAlignment.Center);
		}
	}

	protected override void OnUnfocus()
	{
		_isFocused = false;
		NHoverTipSet.Remove(this);
		_hoverTween?.Kill();
		_hoverTween = CreateTween().SetParallel();
		if (Potion != null)
		{
			if (!_disabledUntilPotionRemoved)
			{
				_hoverTween.TweenProperty(Potion, "scale", _potionScale, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			}
		}
		else
		{
			_hoverTween.TweenProperty(_emptyIcon, "scale", _potionScale, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		}
	}

	protected override void OnPress()
	{
		if (Potion != null)
		{
			GetViewport().SetInputAsHandled();
		}
	}

	protected override void OnRelease()
	{
		if (_isUsable)
		{
			OpenPotionPopup();
		}
		if (Potion != null)
		{
			GetViewport().SetInputAsHandled();
		}
	}

	private void OpenPotionPopup()
	{
		if (HasPotion && !Potion.Model.Owner.RunState.IsGameOver && !_disabledUntilPotionRemoved)
		{
			NHoverTipSet.Remove(this);
			_popup = NPotionPopup.Create(this);
			this.AddChildSafely(_popup);
		}
	}

	public void AddPotion(NPotion potion)
	{
		if (Potion != null)
		{
			throw new InvalidOperationException("Slot already contains a potion");
		}
		Potion = potion;
		_emptyPotionTween?.Kill();
		_emptyIcon.Modulate = Colors.Transparent;
		this.AddChildSafely(Potion);
		Potion.Scale = _potionScale;
		Potion.PivotOffset = Potion.Size * 0.5f;
	}

	public void DisableUntilPotionRemoved()
	{
		if (_popup != null && GodotObject.IsInstanceValid(_popup))
		{
			_popup.Remove();
		}
		_disabledUntilPotionRemoved = true;
		TaskHelper.RunSafely(GrayPotionHolderUntilPlayedAfterDelay());
		this.TryGrabFocus();
	}

	private async Task GrayPotionHolderUntilPlayedAfterDelay()
	{
		_cancelGrayOutPotionSource = new CancellationTokenSource();
		await Task.Delay(100, _cancelGrayOutPotionSource.Token);
		if (!_cancelGrayOutPotionSource.IsCancellationRequested)
		{
			base.Modulate = StsColors.gray;
		}
	}

	public void CancelPotionUse()
	{
		_cancelGrayOutPotionSource?.Cancel();
		_disabledUntilPotionRemoved = false;
		base.Modulate = Colors.White;
	}

	public void RemoveUsedPotion()
	{
		if (Potion == null)
		{
			throw new InvalidOperationException("This slot doesn't contain a potion");
		}
		if (_popup != null && GodotObject.IsInstanceValid(_popup))
		{
			_popup.Remove();
		}
		NHoverTipSet.Remove(this);
		_disabledUntilPotionRemoved = false;
		_cancelGrayOutPotionSource?.Cancel();
		base.Modulate = Colors.White;
		NPotion potionToRemove = Potion;
		Potion = null;
		Tween tween = CreateTween();
		tween.TweenProperty(potionToRemove, "scale", Vector2.Zero, 0.20000000298023224).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Back)
			.FromCurrent();
		tween.TweenCallback(Callable.From(delegate
		{
			this.RemoveChildSafely(potionToRemove);
			potionToRemove.QueueFreeSafely();
		}));
		if (base.IsFocused)
		{
			NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, _emptyHoverTip);
			nHoverTipSet.GlobalPosition = base.GlobalPosition + Vector2.Down * base.Size.Y * 1.5f;
			nHoverTipSet.SetAlignment(this, HoverTipAlignment.Center);
		}
		_emptyPotionTween?.Kill();
		_emptyPotionTween = CreateTween();
		_emptyPotionTween.TweenProperty(_emptyIcon, "modulate", Colors.White, 0.20000000298023224).SetDelay(0.20000000298023224);
	}

	public void DiscardPotion()
	{
		if (Potion == null)
		{
			throw new InvalidOperationException("This slot doesn't contain a potion");
		}
		if (_popup != null && GodotObject.IsInstanceValid(_popup))
		{
			_popup.Remove();
		}
		_disabledUntilPotionRemoved = false;
		_cancelGrayOutPotionSource?.Cancel();
		base.Modulate = Colors.White;
		NPotion potionToRemove = Potion;
		Potion = null;
		Tween tween = CreateTween();
		tween.TweenProperty(potionToRemove, "position:y", -100f, 0.4000000059604645).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Back);
		tween.TweenCallback(Callable.From(delegate
		{
			this.RemoveChildSafely(potionToRemove);
			potionToRemove.QueueFreeSafely();
		}));
		_emptyPotionTween?.Kill();
		_emptyPotionTween = CreateTween();
		_emptyPotionTween.TweenProperty(_emptyIcon, "modulate", Colors.White, 0.20000000298023224).FromCurrent().SetDelay(0.20000000298023224);
	}

	public async Task UsePotion()
	{
		if (Potion == null)
		{
			Log.Warn("Tried to use potion in holder, but potion node is null!");
			return;
		}
		TargetType targetType = Potion.Model.TargetType;
		bool flag = ((targetType == TargetType.AnyEnemy || targetType == TargetType.TargetedNoCreature) ? true : false);
		if (flag || Potion.Model.CanThrowAtAlly())
		{
			RunManager.Instance.HoveredModelTracker.OnLocalPotionSelected(Potion.Model);
			await TargetNode(Potion.Model.TargetType);
			RunManager.Instance.HoveredModelTracker.OnLocalPotionDeselected();
		}
		else
		{
			Potion.Model.EnqueueManualUse(Potion.Model.Owner.Creature);
			this.TryGrabFocus();
		}
	}

	private async Task TargetNode(TargetType targetType)
	{
		Vector2 startPosition = base.GlobalPosition + Vector2.Right * base.Size.X * 0.5f + Vector2.Down * 50f;
		NTargetManager instance = NTargetManager.Instance;
		bool isUsingController = NControllerManager.Instance.IsUsingController;
		instance.StartTargeting(targetType, startPosition, isUsingController ? TargetMode.Controller : TargetMode.ClickMouseToTarget, ShouldCancelTargeting, null);
		Creature creature = Potion.Model.Owner.Creature;
		if (isUsingController && CombatManager.Instance.IsInProgress)
		{
			CombatState combatState = creature.CombatState;
			List<Creature> source = (targetType switch
			{
				TargetType.AnyEnemy => combatState.GetOpponentsOf(creature), 
				TargetType.AnyPlayer => combatState.GetTeammatesOf(creature), 
				_ => throw new ArgumentOutOfRangeException("targetType", targetType, null), 
			}).Where((Creature c) => c.IsAlive).ToList();
			NCombatRoom.Instance.RestrictControllerNavigation(source.Select((Creature c) => NCombatRoom.Instance.GetCreatureNode(c).Hitbox));
			NCombatRoom.Instance.GetCreatureNode(source.First()).Hitbox.TryGrabFocus();
		}
		else if (isUsingController && targetType == TargetType.AnyPlayer)
		{
			NMultiplayerPlayerStateContainer multiplayerPlayerContainer = NRun.Instance.GlobalUi.MultiplayerPlayerContainer;
			multiplayerPlayerContainer.FirstPlayerState?.Hitbox.TryGrabFocus();
			multiplayerPlayerContainer.LockNavigation();
		}
		bool flag = Potion.Model is FoulPotion;
		bool flag2 = creature.Player.RunState.CurrentRoom.RoomType == RoomType.Shop;
		bool isFoulPotionInShop = isUsingController && flag && flag2;
		if (isFoulPotionInShop)
		{
			NMerchantButton merchantButton = NMerchantRoom.Instance.MerchantButton;
			merchantButton.SetFocusMode(FocusModeEnum.All);
			merchantButton.TryGrabFocus();
		}
		try
		{
			Node node = await instance.SelectionFinished();
			NCombatRoom.Instance?.EnableControllerNavigation();
			NRun.Instance.GlobalUi.MultiplayerPlayerContainer.UnlockNavigation();
			if (node != null)
			{
				Creature creature2;
				if (!(node is NCreature nCreature))
				{
					if (!(node is NMultiplayerPlayerState nMultiplayerPlayerState))
					{
						if (!(node is NMerchantButton))
						{
							throw new ArgumentOutOfRangeException("targetNode", node, null);
						}
						creature2 = null;
					}
					else
					{
						creature2 = nMultiplayerPlayerState.Player.Creature;
					}
				}
				else
				{
					creature2 = nCreature.Entity;
				}
				Creature target = creature2;
				Potion.Model.EnqueueManualUse(target);
			}
		}
		finally
		{
			if (isFoulPotionInShop)
			{
				NMerchantButton merchantButton2 = NMerchantRoom.Instance.MerchantButton;
				merchantButton2.SetFocusMode(FocusModeEnum.None);
			}
		}
		this.TryGrabFocus();
	}

	private bool ShouldCancelTargeting()
	{
		if (Potion != null)
		{
			if (CombatManager.Instance.IsInProgress)
			{
				if (NOverlayStack.Instance.ScreenCount <= 0)
				{
					return NCapstoneContainer.Instance.InUse;
				}
				return true;
			}
			return false;
		}
		return true;
	}

	public async Task ShineOnStartOfCombat()
	{
		if (HasPotion)
		{
			Potion.DoBounce();
			await Cmd.Wait(0.25f);
			NDebugAudioManager.Instance?.Play(Rng.Chaotic.NextItem(TmpSfx.PotionSlosh), 0.3f, PitchVariance.Large);
		}
	}
}
