using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NKnowledgeDemonVfx : Node
{
	private Node2D _fireNode1;

	private Node2D _fireNode2;

	private Node2D _fireNode3;

	private Node2D _fireNode4;

	private GpuParticles2D _explosionParticles;

	private GpuParticles2D _damageParticles;

	private GpuParticles2D _emberParticles;

	private GpuParticles2D _thinEmberParticles;

	private Node2D _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_fireNode1 = _parent.GetNode<Node2D>("FireSlot1/FireHolder1");
		_fireNode2 = _parent.GetNode<Node2D>("FireSlot2/FireHolder2");
		_fireNode3 = _parent.GetNode<Node2D>("FireSlot3/FireHolder3");
		_fireNode4 = _parent.GetNode<Node2D>("FireSlot4/FireHolder4");
		_explosionParticles = _parent.GetNode<GpuParticles2D>("ExplosionParticles");
		_damageParticles = _parent.GetNode<GpuParticles2D>("DamageParticles");
		_emberParticles = _parent.GetNode<GpuParticles2D>("EmberParticles");
		_thinEmberParticles = _parent.GetNode<GpuParticles2D>("ThinEmberParticles");
		_fireNode1.Visible = false;
		_fireNode2.Visible = false;
		_fireNode3.Visible = false;
		_fireNode4.Visible = false;
		_explosionParticles.Emitting = false;
		_explosionParticles.OneShot = true;
		_damageParticles.Emitting = false;
		_damageParticles.OneShot = true;
		_emberParticles.Emitting = false;
		_thinEmberParticles.Emitting = false;
		OnBurningEnd();
		_animController.GetAnimationState().SetAnimation("idle_loop");
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		string eventName = new MegaEvent(spineEvent).GetData().GetEventName();
		if (eventName == null)
		{
			return;
		}
		switch (eventName.Length)
		{
		case 11:
			switch (eventName[0])
			{
			case 't':
				if (eventName == "take_damage")
				{
					OnTakeDamage();
				}
				break;
			case 'b':
				if (eventName == "burning_end")
				{
					OnBurningEnd();
				}
				break;
			}
			break;
		case 7:
			if (eventName == "explode")
			{
				OnExplode();
			}
			break;
		case 13:
			if (eventName == "burning_start")
			{
				OnBurningStart();
			}
			break;
		case 12:
			if (eventName == "embers_start")
			{
				OnEmbersStart();
			}
			break;
		case 17:
			if (eventName == "thin_embers_start")
			{
				OnThinEmbersStart();
			}
			break;
		case 10:
			if (eventName == "embers_end")
			{
				OnEmbersEnd();
			}
			break;
		case 15:
			if (eventName == "thin_embers_end")
			{
				OnThinEmbersEnd();
			}
			break;
		case 8:
		case 9:
		case 14:
		case 16:
			break;
		}
	}

	private void OnExplode()
	{
		_explosionParticles.Restart();
	}

	private void OnTakeDamage()
	{
		_damageParticles.Restart();
	}

	private void OnBurningStart()
	{
		_fireNode1.Visible = true;
		_fireNode2.Visible = true;
		_fireNode3.Visible = true;
		_fireNode4.Visible = true;
	}

	private void OnEmbersStart()
	{
		_emberParticles.Restart();
	}

	private void OnThinEmbersStart()
	{
		_thinEmberParticles.Restart();
	}

	private void OnBurningEnd()
	{
		_fireNode1.Visible = false;
		_fireNode2.Visible = false;
		_fireNode3.Visible = false;
		_fireNode4.Visible = false;
	}

	private void OnEmbersEnd()
	{
		_emberParticles.Emitting = false;
	}

	private void OnThinEmbersEnd()
	{
		_thinEmberParticles.Emitting = false;
	}
}
