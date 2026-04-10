using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Vfx;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NControllerCardPlay : NCardPlay
{
	[Signal]
	public delegate void ConfirmedEventHandler();

	[Signal]
	public delegate void CanceledEventHandler();

	public override void _Input(InputEvent inputEvent)
	{
		if (inputEvent is InputEventAction inputEventAction)
		{
			if (inputEventAction.IsActionPressed(MegaInput.select))
			{
				EmitSignal(SignalName.Confirmed);
			}
			if (inputEventAction.IsActionPressed(MegaInput.cancel))
			{
				EmitSignal(SignalName.Canceled);
			}
		}
	}

	public static NControllerCardPlay Create(NHandCardHolder holder)
	{
		NControllerCardPlay nControllerCardPlay = new NControllerCardPlay();
		nControllerCardPlay.Holder = holder;
		nControllerCardPlay.Player = holder.CardModel.Owner;
		return nControllerCardPlay;
	}

	public override void Start()
	{
		if (base.Card == null || base.CardNode == null)
		{
			return;
		}
		NDebugAudioManager.Instance?.Play("card_select.mp3");
		NHoverTipSet.Remove(base.Holder);
		if (!base.Card.CanPlay(out UnplayableReason reason, out AbstractModel preventer))
		{
			CannotPlayThisCardFtueCheck(base.Card);
			CancelPlayCard();
			LocString playerDialogueLine = reason.GetPlayerDialogueLine(preventer);
			if (playerDialogueLine != null)
			{
				NCombatRoom.Instance.CombatVfxContainer.AddChildSafely(NThoughtBubbleVfx.Create(playerDialogueLine.GetFormattedText(), base.Card.Owner.Creature, 1.0));
			}
			return;
		}
		TryShowEvokingOrbs();
		base.CardNode.CardHighlight.AnimFlash();
		CenterCard();
		TargetType targetType = base.Card.TargetType;
		if ((targetType == TargetType.AnyEnemy || targetType == TargetType.AnyAlly) ? true : false)
		{
			TaskHelper.RunSafely(SingleCreatureTargeting(base.Card.TargetType));
		}
		else
		{
			MultiCreatureTargeting();
		}
	}

	private async Task SingleCreatureTargeting(TargetType targetType)
	{
		NTargetManager targetManager = NTargetManager.Instance;
		targetManager.Connect(NTargetManager.SignalName.CreatureHovered, Callable.From<NCreature>(base.OnCreatureHover));
		targetManager.Connect(NTargetManager.SignalName.CreatureUnhovered, Callable.From<NCreature>(base.OnCreatureUnhover));
		targetManager.StartTargeting(targetType, base.CardNode, TargetMode.Controller, () => !GodotObject.IsInstanceValid(this) || !NControllerManager.Instance.IsUsingController, null);
		Creature owner = base.Card.Owner.Creature;
		List<Creature> list = new List<Creature>();
		switch (targetType)
		{
		case TargetType.AnyEnemy:
			list = (from c in owner.CombatState.GetOpponentsOf(owner)
				where c.IsHittable
				select c).ToList();
			break;
		case TargetType.AnyAlly:
			list = base.Card.CombatState.PlayerCreatures.Where((Creature c) => c.IsHittable && c != owner).ToList();
			break;
		}
		if (list.Count == 0)
		{
			CancelPlayCard();
			return;
		}
		NCombatRoom.Instance.RestrictControllerNavigation(list.Select((Creature c) => NCombatRoom.Instance.GetCreatureNode(c).Hitbox));
		NCombatRoom.Instance.GetCreatureNode(list.First()).Hitbox.TryGrabFocus();
		NCreature nCreature = (NCreature)(await targetManager.SelectionFinished());
		if (GodotObject.IsInstanceValid(this))
		{
			targetManager.Disconnect(NTargetManager.SignalName.CreatureHovered, Callable.From<NCreature>(base.OnCreatureHover));
			targetManager.Disconnect(NTargetManager.SignalName.CreatureUnhovered, Callable.From<NCreature>(base.OnCreatureUnhover));
			if (nCreature != null)
			{
				TryPlayCard(nCreature.Entity);
			}
			else
			{
				CancelPlayCard();
			}
		}
	}

	private void MultiCreatureTargeting()
	{
		NCombatRoom.Instance.RestrictControllerNavigation(Array.Empty<Control>());
		ShowMultiCreatureTargetingVisuals();
		Connect(SignalName.Confirmed, Callable.From(delegate
		{
			TryPlayCard(null);
		}));
		Connect(SignalName.Canceled, Callable.From(base.CancelPlayCard));
	}

	protected override void OnCancelPlayCard()
	{
		base.Holder.TryGrabFocus();
	}

	protected override void Cleanup()
	{
		base.Cleanup();
		NCombatRoom.Instance.EnableControllerNavigation();
		NCombatRoom.Instance.Ui.Hand.DefaultFocusedControl.TryGrabFocus();
	}
}
