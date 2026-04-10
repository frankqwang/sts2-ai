using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NAmalgamVfx : Node
{
	[Export(PropertyHint.None, "")]
	private GpuParticles2D _hitFxParticles;

	[Export(PropertyHint.None, "")]
	private Node2D _hitBoneNode;

	private CpuParticles2D _deathBodyParticles;

	private GpuParticles2D _laserBaseParticles;

	private GpuParticles2D _hitParticles1;

	private GpuParticles2D _hitParticles2;

	private GpuParticles2D _hitParticles3;

	private GpuParticles2D _constantSparks1;

	private GpuParticles2D _constantSparks2;

	private GpuParticles2D _constantSparks3;

	private Node2D _torch1Node;

	private Node2D _torch2Node;

	private Node2D _torch3Node;

	private Node _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_deathBodyParticles = _parent.GetNode<CpuParticles2D>("CPUDeathParticles");
		_deathBodyParticles.Emitting = false;
		_deathBodyParticles.OneShot = true;
		_laserBaseParticles = _parent.GetNode<GpuParticles2D>("laserBaseBone/laserBaseParticles");
		_laserBaseParticles.Emitting = false;
		_torch1Node = _parent.GetNode<Node2D>("torch1Slot/fire1_small_green");
		_torch1Node.Visible = true;
		_torch2Node = _parent.GetNode<Node2D>("torch2Slot/fire2_small_green");
		_torch2Node.Visible = true;
		_torch3Node = _parent.GetNode<Node2D>("torch3Slot/fire3_small_green");
		_torch3Node.Visible = true;
		_hitParticles1 = _parent.GetNode<GpuParticles2D>("torch1UnscaledBone/hitParticles");
		_hitParticles1.Emitting = false;
		_hitParticles1.OneShot = true;
		_hitParticles2 = _parent.GetNode<GpuParticles2D>("torch2UnscaledBone/hitParticles");
		_hitParticles2.Emitting = false;
		_hitParticles2.OneShot = true;
		_hitParticles3 = _parent.GetNode<GpuParticles2D>("torch3UnscaledBone/hitParticles");
		_hitParticles3.Emitting = false;
		_hitParticles3.OneShot = true;
		_constantSparks1 = _parent.GetNode<GpuParticles2D>("torch1Slot/constantParticles");
		_constantSparks2 = _parent.GetNode<GpuParticles2D>("torch2Slot/constantParticles");
		_constantSparks3 = _parent.GetNode<GpuParticles2D>("torch3Slot/constantParticles");
		_hitFxParticles.Visible = false;
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
		case 4:
			switch (eventName[3])
			{
			case '1':
				if (eventName == "hit1")
				{
					PlayHit1();
				}
				break;
			case '2':
				if (eventName == "hit2")
				{
					PlayHit2();
				}
				break;
			case '3':
				if (eventName == "hit3")
				{
					PlayHit3();
				}
				break;
			}
			break;
		case 14:
			switch (eventName[6])
			{
			case 'b':
				if (eventName == "laser_base_off")
				{
					PlayLaserBase(starting: false);
				}
				break;
			case 'h':
				if (eventName == "laser_hit_fire")
				{
					PlayLaserHit(starting: true);
				}
				break;
			}
			break;
		case 7:
			if (eventName == "go_poof")
			{
				PoofToDeath();
			}
			break;
		case 11:
			if (eventName == "torches_out")
			{
				KillTorches();
			}
			break;
		case 10:
			if (eventName == "torches_on")
			{
				RestartTorches();
			}
			break;
		case 15:
			if (eventName == "laser_base_fire")
			{
				PlayLaserBase(starting: true);
			}
			break;
		case 13:
			if (eventName == "laser_hit_off")
			{
				PlayLaserHit(starting: false);
			}
			break;
		case 5:
		case 6:
		case 8:
		case 9:
		case 12:
			break;
		}
	}

	private void PoofToDeath()
	{
		_deathBodyParticles.Restart();
	}

	private void RestartTorches()
	{
		_torch1Node.Visible = true;
		_torch2Node.Visible = true;
		_torch3Node.Visible = true;
		_constantSparks1.Emitting = true;
		_constantSparks2.Emitting = true;
		_constantSparks3.Emitting = true;
	}

	private void KillTorches()
	{
		_torch1Node.Visible = false;
		_torch2Node.Visible = false;
		_torch3Node.Visible = false;
		_constantSparks1.Emitting = false;
		_constantSparks2.Emitting = false;
		_constantSparks3.Emitting = false;
	}

	private void PlayHit1()
	{
		_hitParticles1.Restart();
	}

	private void PlayHit2()
	{
		_hitParticles2.Restart();
	}

	private void PlayHit3()
	{
		_hitParticles3.Restart();
	}

	private void PlayLaserBase(bool starting)
	{
		if (starting)
		{
			_laserBaseParticles.Visible = true;
			_laserBaseParticles.Restart();
		}
		else
		{
			_laserBaseParticles.Emitting = false;
			_laserBaseParticles.Visible = false;
		}
	}

	private void PlayLaserHit(bool starting)
	{
		if (starting)
		{
			_hitFxParticles.GlobalPosition = _hitBoneNode.GlobalPosition;
			_hitFxParticles.Visible = true;
			_hitFxParticles.Restart();
		}
		else
		{
			_hitFxParticles.Emitting = false;
			_hitFxParticles.Visible = false;
		}
	}
}
