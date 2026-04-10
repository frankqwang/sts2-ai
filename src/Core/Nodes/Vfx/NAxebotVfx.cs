using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NAxebotVfx : Node
{
	private GpuParticles2D _hurtParticles1;

	private GpuParticles2D _hurtParticles2;

	private GpuParticles2D _smokeParticlesLeft;

	private GpuParticles2D _smokeParticlesRight;

	private Node2D _parent;

	private MegaSprite _animController;

	private int _currentWeapon = 1;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_hurtParticles1 = _parent.GetNode<GpuParticles2D>("SparksBoneNode/HurtParticles1");
		_hurtParticles2 = _parent.GetNode<GpuParticles2D>("SparksBoneNode/HurtParticles2");
		_smokeParticlesLeft = _parent.GetNode<GpuParticles2D>("SmokeNodeLeft/SmokeParticles");
		_smokeParticlesRight = _parent.GetNode<GpuParticles2D>("SmokeNodeRight/SmokeParticles");
		_hurtParticles1.OneShot = true;
		_hurtParticles2.OneShot = true;
		_hurtParticles1.Emitting = false;
		_hurtParticles2.Emitting = false;
		_smokeParticlesLeft.Emitting = false;
		_smokeParticlesRight.Emitting = false;
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		switch (new MegaEvent(spineEvent).GetData().GetEventName())
		{
		case "start_hurt_sparks":
			TurnOnHurt();
			break;
		case "start_death_sparks1":
			TurnOnDeath1();
			break;
		case "start_death_sparks2":
			TurnOnDeath2();
			break;
		case "landing_smoke_start":
			TurnOnLandingSmoke();
			break;
		case "landing_smoke_end":
			TurnOffLandingSmoke();
			break;
		}
	}

	private void TurnOnDeath1()
	{
		_hurtParticles1.Restart();
	}

	private void TurnOnDeath2()
	{
		_hurtParticles2.Restart();
	}

	private void TurnOnHurt()
	{
		if (_currentWeapon == 1)
		{
			_hurtParticles1.Restart();
			_currentWeapon = 2;
		}
		else
		{
			_hurtParticles2.Restart();
			_currentWeapon = 1;
		}
	}

	private void TurnOnLandingSmoke()
	{
		_smokeParticlesLeft.Restart();
		_smokeParticlesRight.Restart();
	}

	private void TurnOffLandingSmoke()
	{
		_smokeParticlesLeft.Emitting = false;
		_smokeParticlesRight.Emitting = false;
	}
}
