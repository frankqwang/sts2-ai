using System;
using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Events;

public partial class NCombatEventLayout : NEventLayout
{
	public const string combatScenePath = "res://scenes/events/combat_event_layout.tscn";

	private Control _combatRoomContainer;

	public NCombatRoom? EmbeddedCombatRoom { get; private set; }

	public bool HasCombatStarted { get; private set; }

	public override Control? DefaultFocusedControl
	{
		get
		{
			if (!HasCombatStarted)
			{
				return base.DefaultFocusedControl;
			}
			return EmbeddedCombatRoom?.DefaultFocusedControl;
		}
	}

	public override void _Ready()
	{
		base._Ready();
		_combatRoomContainer = GetNode<Control>("%CombatRoomContainer");
	}

	public void SetCombatRoomNode(NCombatRoom? combatRoomNode)
	{
		if (combatRoomNode != null)
		{
			if (EmbeddedCombatRoom != null)
			{
				throw new InvalidOperationException("Combat room node was already set.");
			}
			EmbeddedCombatRoom = combatRoomNode;
			_combatRoomContainer.AddChildSafely(combatRoomNode);
		}
	}

	public override void SetEvent(EventModel eventModel)
	{
		IRunState runState = eventModel.Owner.RunState;
		ICombatRoomVisuals visuals = eventModel.CreateCombatRoomVisuals(runState.Players, runState.Act);
		NCombatRoom nCombatRoom = NCombatRoom.Create(visuals, CombatRoomMode.VisualOnly);
		SetCombatRoomNode(nCombatRoom);
		nCombatRoom?.SetUpBackground(runState);
		base.SetEvent(eventModel);
	}

	protected override void InitializeVisuals()
	{
	}

	public void HideEventVisuals()
	{
		if (_description != null)
		{
			_description.Visible = false;
		}
		if (_sharedEventLabel != null)
		{
			_sharedEventLabel.Visible = false;
		}
		_optionsContainer.Visible = false;
		HasCombatStarted = true;
		DefaultFocusedControl?.TryGrabFocus();
	}
}
