using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NMyteVfx : Node
{
	private const float _attackHeight = 150f;

	private Node2D _parent;

	private MegaSprite _animController;

	private Node2D _targetBone;

	private Creature? _target;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_targetBone = _parent.GetNode<Node2D>("TargetBone");
		_animController.GetAnimationState().SetAnimation("cast");
	}

	public void SetTarget(Creature? target)
	{
		_target = target;
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		string eventName = new MegaEvent(spineEvent).GetData().GetEventName();
		if (eventName == "start_cast")
		{
			StartCast();
		}
	}

	private void StartCast()
	{
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(_target);
		if (nCreature != null)
		{
			_targetBone.GlobalPosition = new Vector2(nCreature.GlobalPosition.X, nCreature.GlobalPosition.Y - 150f);
		}
	}
}
