using System;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Multiplayer;
using MegaCrit.Sts2.Core.Nodes.RestSite;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NTargetManager : Node2D
{
	[Signal]
	public delegate void CreatureHoveredEventHandler(NCreature creature);

	[Signal]
	public delegate void CreatureUnhoveredEventHandler(NCreature creature);

	[Signal]
	public delegate void NodeHoveredEventHandler(Node node);

	[Signal]
	public delegate void NodeUnhoveredEventHandler(Node node);

	[Signal]
	public delegate void TargetingBeganEventHandler();

	[Signal]
	public delegate void TargetingEndedEventHandler();

	private NTargetingArrow _targetingArrow;

	private TaskCompletionSource<Node?>? _completionSource;

	private Func<bool>? _exitEarlyCondition;

	private Func<Node, bool>? _nodeFilter;

	private TargetMode _targetMode;

	private TargetType _validTargetsType;

	public static NTargetManager Instance => NRun.Instance.GlobalUi.TargetManager;

	public bool IsInSelection => _targetMode != TargetMode.None;

	private Node? HoveredNode { get; set; }

	public long LastTargetingFinishedFrame { get; set; }

	public override void _Ready()
	{
		_targetingArrow = GetNode<NTargetingArrow>("TargetingArrow");
	}

	public override void _EnterTree()
	{
		base._EnterTree();
		if (NControllerManager.Instance != null)
		{
			NControllerManager.Instance.Connect(NControllerManager.SignalName.MouseDetected, Callable.From(CancelTargeting));
			NControllerManager.Instance.Connect(NControllerManager.SignalName.ControllerDetected, Callable.From(CancelTargeting));
		}
		CombatManager.Instance.CombatEnded += OnCombatEnded;
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		if (NControllerManager.Instance != null)
		{
			NControllerManager.Instance.Disconnect(NControllerManager.SignalName.MouseDetected, Callable.From(CancelTargeting));
			NControllerManager.Instance.Disconnect(NControllerManager.SignalName.ControllerDetected, Callable.From(CancelTargeting));
		}
		CombatManager.Instance.CombatEnded -= OnCombatEnded;
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (!IsInSelection)
		{
			return;
		}
		bool flag = false;
		bool cancel = false;
		if (inputEvent is InputEventMouseButton { ButtonIndex: var buttonIndex } inputEventMouseButton)
		{
			switch (buttonIndex)
			{
			case MouseButton.Left:
				if (!inputEventMouseButton.IsReleased())
				{
					break;
				}
				switch (_targetMode)
				{
				case TargetMode.ReleaseMouseToTarget:
					if (HoveredNode != null)
					{
						flag = true;
					}
					else
					{
						_targetMode = TargetMode.ClickMouseToTarget;
					}
					break;
				case TargetMode.ClickMouseToTarget:
					flag = true;
					break;
				}
				break;
			case MouseButton.Right:
				flag = inputEventMouseButton.IsPressed();
				cancel = true;
				break;
			}
		}
		else if (inputEvent.IsActionPressed(MegaInput.select) && HoveredNode != null)
		{
			flag = true;
			cancel = false;
			GetViewport().SetInputAsHandled();
		}
		else if (inputEvent.IsActionPressed(MegaInput.cancel) || inputEvent.IsActionPressed(MegaInput.topPanel))
		{
			flag = true;
			cancel = true;
			GetViewport().SetInputAsHandled();
		}
		if (_exitEarlyCondition != null && _exitEarlyCondition())
		{
			flag = true;
			cancel = true;
		}
		if (flag)
		{
			FinishTargeting(cancel);
		}
	}

	public override void _Process(double delta)
	{
		if (_exitEarlyCondition != null && _exitEarlyCondition())
		{
			FinishTargeting(cancel: true);
		}
		if (HoveredNode is NCreature nCreature)
		{
			Creature entity = nCreature.Entity;
			if (entity != null && !entity.IsHittable && _targetMode == TargetMode.Controller)
			{
				FinishTargeting(cancel: true);
			}
		}
	}

	private void OnCombatEnded(CombatRoom _)
	{
		if (_exitEarlyCondition != null)
		{
			FinishTargeting(cancel: true);
		}
	}

	public void CancelTargeting()
	{
		if (_targetMode != TargetMode.None)
		{
			FinishTargeting(cancel: true);
		}
	}

	private void FinishTargeting(bool cancel)
	{
		NHoverTipSet.shouldBlockHoverTips = false;
		_exitEarlyCondition = null;
		_completionSource.SetResult(cancel ? null : HoveredNode);
		LastTargetingFinishedFrame = GetTree().GetFrame();
		EmitSignal(SignalName.TargetingEnded);
		_targetMode = TargetMode.None;
		_targetingArrow.StopDrawing();
		if (HoveredNode is NCreature nCreature)
		{
			nCreature.HideMultiselectReticle();
		}
		else if (HoveredNode is NRestSiteCharacter nRestSiteCharacter)
		{
			nRestSiteCharacter.Deselect();
		}
		HoveredNode = null;
		RunManager.Instance.InputSynchronizer.SyncLocalIsTargeting(isTargeting: false);
	}

	public async Task<Node?> SelectionFinished()
	{
		return await _completionSource.Task;
	}

	public void StartTargeting(TargetType validTargetsType, Vector2 startPosition, TargetMode startingMode, Func<bool>? exitEarlyCondition, Func<Node, bool>? nodeFilter)
	{
		if (!validTargetsType.IsSingleTarget())
		{
			throw new InvalidOperationException($"Tried to begin targeting with invalid ActionTarget {validTargetsType}!");
		}
		_validTargetsType = validTargetsType;
		_targetingArrow.StartDrawingFrom(startPosition, startingMode == TargetMode.Controller);
		_completionSource = new TaskCompletionSource<Node>();
		_exitEarlyCondition = exitEarlyCondition;
		_nodeFilter = nodeFilter;
		_targetMode = startingMode;
		NHoverTipSet.shouldBlockHoverTips = true;
		EmitSignal(SignalName.TargetingBegan);
		RunManager.Instance.InputSynchronizer.SyncLocalIsTargeting(isTargeting: true);
		foreach (NCreature item in NCombatRoom.Instance?.CreatureNodes ?? Array.Empty<NCreature>())
		{
			item.OnTargetingStarted();
		}
	}

	public void StartTargeting(TargetType validTargetsType, Control control, TargetMode startingMode, Func<bool>? exitEarlyCondition, Func<Node, bool>? nodeFilter)
	{
		if (!validTargetsType.IsSingleTarget())
		{
			throw new InvalidOperationException($"Tried to begin targeting with invalid ActionTarget {validTargetsType}!");
		}
		_validTargetsType = validTargetsType;
		_targetingArrow.StartDrawingFrom(control, startingMode == TargetMode.Controller);
		_completionSource = new TaskCompletionSource<Node>();
		_exitEarlyCondition = exitEarlyCondition;
		_nodeFilter = nodeFilter;
		_targetMode = startingMode;
		NHoverTipSet.shouldBlockHoverTips = true;
		EmitSignal(SignalName.TargetingBegan);
		RunManager.Instance.InputSynchronizer.SyncLocalIsTargeting(isTargeting: true);
		foreach (NCreature item in NCombatRoom.Instance?.CreatureNodes ?? Array.Empty<NCreature>())
		{
			item.OnTargetingStarted();
		}
	}

	public bool AllowedToTargetNode(Node node)
	{
		if (_nodeFilter != null && !_nodeFilter(node))
		{
			return false;
		}
		if (node is NCreature nCreature)
		{
			return AllowedToTargetCreature(nCreature.Entity);
		}
		if (node is NMultiplayerPlayerState nMultiplayerPlayerState)
		{
			return AllowedToTargetCreature(nMultiplayerPlayerState.Player.Creature);
		}
		return true;
	}

	private bool AllowedToTargetCreature(Creature creature)
	{
		switch (_validTargetsType)
		{
		case TargetType.AnyEnemy:
			if (creature.Side != CombatSide.Enemy)
			{
				return false;
			}
			break;
		case TargetType.AnyPlayer:
			if (!creature.IsPlayer || creature.IsDead)
			{
				return false;
			}
			break;
		case TargetType.AnyAlly:
			if (!creature.IsPlayer || creature.IsDead)
			{
				return false;
			}
			if (LocalContext.IsMe(creature.Player))
			{
				return false;
			}
			break;
		default:
			throw new ArgumentOutOfRangeException("_validTargetsType", _validTargetsType, null);
		}
		return true;
	}

	public void OnNodeHovered(Node node)
	{
		if (!IsInSelection || !AllowedToTargetNode(node))
		{
			return;
		}
		if (node is NCreature creature)
		{
			OnCreatureHovered(creature);
			return;
		}
		HoveredNode = node;
		_targetingArrow.SetHighlightingOn(isEnemy: false);
		if (_targetMode == TargetMode.Controller)
		{
			if (!(node is NMultiplayerPlayerState nMultiplayerPlayerState))
			{
				if (!(node is Control control))
				{
					if (node is Node2D node2D)
					{
						_targetingArrow.UpdateDrawingTo(node2D.GlobalPosition);
					}
				}
				else
				{
					_targetingArrow.UpdateDrawingTo(control.GlobalPosition + control.PivotOffset);
				}
			}
			else
			{
				_targetingArrow.UpdateDrawingTo(nMultiplayerPlayerState.GlobalPosition + Vector2.Right * nMultiplayerPlayerState.Hitbox.Size.X + Vector2.Down * nMultiplayerPlayerState.Hitbox.Size.Y / 2f);
			}
		}
		EmitSignal(SignalName.NodeHovered, node);
	}

	public void OnNodeUnhovered(Node node)
	{
		if (IsInSelection && AllowedToTargetNode(node))
		{
			if (node is NCreature creature)
			{
				OnCreatureUnhovered(creature);
				return;
			}
			HoveredNode = null;
			_targetingArrow.SetHighlightingOff();
			EmitSignal(SignalName.NodeUnhovered, node);
		}
	}

	private void OnCreatureHovered(NCreature creature)
	{
		if (Hook.ShouldAllowTargeting(creature.Entity.CombatState, creature.Entity, out AbstractModel preventer))
		{
			HoveredNode = creature;
			_targetingArrow.SetHighlightingOn(creature.Entity.IsEnemy);
			creature.ShowSingleSelectReticle();
			EmitSignal(SignalName.CreatureHovered, creature);
			if (_targetMode == TargetMode.Controller)
			{
				_targetingArrow.UpdateDrawingTo(creature.VfxSpawnPosition);
			}
		}
		else
		{
			TaskHelper.RunSafely(preventer.AfterTargetingBlockedVfx(creature.Entity));
		}
	}

	private void OnCreatureUnhovered(NCreature creature)
	{
		EmitSignal(SignalName.CreatureUnhovered, creature);
		if (HoveredNode == creature)
		{
			HoveredNode = null;
		}
		_targetingArrow.SetHighlightingOff();
		if (_targetMode != TargetMode.None)
		{
			creature.HideSingleSelectReticle();
		}
	}
}
