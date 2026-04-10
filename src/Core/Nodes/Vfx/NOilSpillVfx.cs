using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NOilSpillVfx : Node
{
	private const int _slamSprayAmount = 800;

	private const int _sprayAttackAmount = 2000;

	private const int _deathSprayAmount = 500;

	private const float _slamSprayLifetime = 0.75f;

	private GpuParticles2D _droolParticles;

	private GpuParticles2D _sprayParticles;

	private GpuParticles2D _rainDropParticles;

	private Node2D _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_animController.ConnectAnimationStarted(Callable.From<GodotObject, GodotObject, GodotObject>(OnAnimationStart));
		_droolParticles = _parent.GetNode<GpuParticles2D>("MouthDribbleBoneNode/DribbleParticles");
		_sprayParticles = _parent.GetNode<GpuParticles2D>("MouthSpraySlot/SprayParticles");
		_rainDropParticles = _parent.GetNode<GpuParticles2D>("MouthSpraySlot/RainDropParticles");
		_rainDropParticles.OneShot = true;
		_droolParticles.Restart();
		_rainDropParticles.Restart();
		_sprayParticles.Restart();
		TurnOffSprayAttack();
		TurnOffSlamSpray();
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
			case 's':
				if (eventName == "spray_start")
				{
					TurnOnSprayAttack();
				}
				break;
			case 'd':
				if (eventName == "drool_start")
				{
					TurnOnDrool();
				}
				break;
			}
			break;
		case 9:
			switch (eventName[0])
			{
			case 's':
				if (eventName == "spray_end")
				{
					TurnOffSprayAttack();
				}
				break;
			case 'd':
				if (eventName == "drool_end")
				{
					TurnOffDrool();
				}
				break;
			}
			break;
		case 16:
			if (eventName == "slam_spray_start")
			{
				TurnOnSlamSpray();
			}
			break;
		case 14:
			if (eventName == "slam_spray_end")
			{
				TurnOffSlamSpray();
			}
			break;
		case 17:
			if (eventName == "death_spray_start")
			{
				TurnOnDeathSpray();
			}
			break;
		case 15:
			if (eventName == "death_spray_end")
			{
				TurnOffDeathSpray();
			}
			break;
		case 10:
		case 12:
		case 13:
			break;
		}
	}

	private void OnAnimationStart(GodotObject spineSprite, GodotObject animationState, GodotObject trackEntry)
	{
		string name = new MegaAnimationState(animationState).GetCurrent(0).GetAnimation().GetName();
		if (name != "slam")
		{
			TurnOffSprayAttack();
		}
		if (name != "spray")
		{
			TurnOffSlamSpray();
		}
	}

	private void TurnOnSprayAttack()
	{
		_sprayParticles.Amount = 2000;
		_sprayParticles.Emitting = true;
	}

	private void TurnOffSprayAttack()
	{
		_sprayParticles.Emitting = false;
	}

	private void TurnOnSlamSpray()
	{
		_rainDropParticles.OneShot = false;
		_rainDropParticles.Amount = 800;
		_rainDropParticles.Explosiveness = 0f;
		_rainDropParticles.Lifetime = 0.75;
		_rainDropParticles.Restart();
	}

	private void TurnOffSlamSpray()
	{
		_rainDropParticles.Emitting = false;
	}

	private void TurnOnDeathSpray()
	{
		_sprayParticles.Amount = 500;
		_sprayParticles.Emitting = true;
	}

	private void TurnOffDeathSpray()
	{
		_sprayParticles.Emitting = false;
	}

	private void TurnOnDrool()
	{
		_droolParticles.Emitting = true;
	}

	private void TurnOffDrool()
	{
		_droolParticles.Emitting = false;
	}
}
