using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NDevotedSculptorVfx : Node
{
	private GpuParticles2D _voiceParticles;

	private GpuParticles2D _attackParticles;

	private Node2D _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_voiceParticles = _parent.GetNode<GpuParticles2D>("VoiceBoneNode/VoiceParticles");
		_voiceParticles.Emitting = false;
		_voiceParticles.OneShot = true;
		_attackParticles = _parent.GetNode<GpuParticles2D>("AttackParticles");
		_attackParticles.Emitting = false;
		_attackParticles.OneShot = true;
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		string eventName = new MegaEvent(spineEvent).GetData().GetEventName();
		if (!(eventName == "caw"))
		{
			if (eventName == "attack")
			{
				StartAttack();
			}
		}
		else
		{
			StartVoice();
		}
	}

	private void StartVoice()
	{
		_voiceParticles.Restart();
	}

	private void StartAttack()
	{
		_attackParticles.Restart();
	}
}
