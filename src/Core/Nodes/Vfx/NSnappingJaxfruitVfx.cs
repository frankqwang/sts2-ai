using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NSnappingJaxfruitVfx : Node
{
	private const float _attackHeight = 130f;

	private Creature? _target;

	private Node2D _projectileBone;

	private Node2D _targetBone;

	private GpuParticles2D _glowParticles;

	private GpuParticles2D _blobParticles;

	private NBasicTrail _trail;

	private Node2D _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_projectileBone = _parent.GetNode<Node2D>("ProjectileAttachBone");
		_targetBone = _parent.GetNode<Node2D>("TargetBone");
		_glowParticles = _parent.GetNode<GpuParticles2D>("ProjectileAttachBone/GlowParticles");
		_blobParticles = _parent.GetNode<GpuParticles2D>("ProjectileAttachBone/BlobParticles");
		_trail = _parent.GetNode<NBasicTrail>("ProjectileAttachBone/Trail");
		ResetCast();
		_animController.GetAnimationState().SetAnimation("charged_loop");
	}

	public void SetTarget(Creature? target)
	{
		_target = target;
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		string eventName = new MegaEvent(spineEvent).GetData().GetEventName();
		if (!(eventName == "cast_start"))
		{
			if (eventName == "cast_end")
			{
				ResetCast();
			}
		}
		else
		{
			StartCast();
		}
	}

	private void StartCast()
	{
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(_target);
		if (nCreature != null)
		{
			_targetBone.GlobalPosition = new Vector2(nCreature.GlobalPosition.X, nCreature.GlobalPosition.Y - 130f);
		}
		_projectileBone.Visible = true;
		_trail.ClearPoints();
		_blobParticles.Restart();
		_glowParticles.Restart();
	}

	private void ResetCast()
	{
		_blobParticles.Emitting = false;
		_glowParticles.Emitting = false;
		_projectileBone.Visible = false;
	}
}
