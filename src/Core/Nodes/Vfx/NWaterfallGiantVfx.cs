using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NWaterfallGiantVfx : Node
{
	private GpuParticles2D _steam1Particles;

	private GpuParticles2D _steam2Particles;

	private GpuParticles2D _steam3Particles;

	private GpuParticles2D _steam4Particles;

	private GpuParticles2D _steam5Particles;

	private GpuParticles2D _steam6Particles;

	private GpuParticles2D _steamLeakParticles1;

	private GpuParticles2D _steamLeakParticles2;

	private GpuParticles2D _steamLeakParticles3;

	private GpuParticles2D _mistParticles;

	private GpuParticles2D _mouthParticles;

	private GpuParticles2D _dropletParticles;

	private bool _isDead;

	private Node2D _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_steam1Particles = _parent.GetNode<GpuParticles2D>("SteamSlot1/steamParticles1");
		_steam2Particles = _parent.GetNode<GpuParticles2D>("SteamSlot2/steamParticles2");
		_steam3Particles = _parent.GetNode<GpuParticles2D>("SteamSlot3/steamParticles3");
		_steam4Particles = _parent.GetNode<GpuParticles2D>("SteamSlot4/steamParticles4");
		_steam5Particles = _parent.GetNode<GpuParticles2D>("SteamSlot5/steamParticles5");
		_steam6Particles = _parent.GetNode<GpuParticles2D>("SteamSlot6/steamParticles6");
		_steamLeakParticles1 = _parent.GetNode<GpuParticles2D>("SteamLeakSlot1/steamLeakParticles1");
		_steamLeakParticles2 = _parent.GetNode<GpuParticles2D>("SteamLeakSlot2/steamLeakParticles2");
		_steamLeakParticles3 = _parent.GetNode<GpuParticles2D>("SteamLeakSlot3/steamLeakParticles3");
		_mistParticles = _parent.GetNode<GpuParticles2D>("MistSlot/MistParticles");
		_dropletParticles = _parent.GetNode<GpuParticles2D>("MistSlot/Droplets");
		_mouthParticles = _parent.GetNode<GpuParticles2D>("MouthDropletsSlot/MouthDroplets");
		_steam1Particles.Emitting = false;
		_steam2Particles.Emitting = false;
		_steam3Particles.Emitting = false;
		_steam4Particles.Emitting = false;
		_steam5Particles.Emitting = false;
		_steam6Particles.Emitting = false;
		_steamLeakParticles1.Emitting = false;
		_steamLeakParticles2.Emitting = false;
		_steamLeakParticles3.Emitting = false;
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
		case 13:
			switch (eventName[6])
			{
			case '1':
				if (eventName == "steam_1_start")
				{
					StartSteam1();
				}
				break;
			case '2':
				if (eventName == "steam_2_start")
				{
					StartSteam2();
				}
				break;
			case '3':
				if (eventName == "steam_3_start")
				{
					StartSteam3();
				}
				break;
			case '5':
				if (eventName == "steam_5_start")
				{
					StartSteam5();
				}
				break;
			case 'a':
				if (eventName == "waterfall_end")
				{
					EndWaterfall();
				}
				break;
			}
			break;
		case 11:
			switch (eventName[6])
			{
			case '1':
				if (eventName == "steam_1_end")
				{
					EndSteam1();
				}
				break;
			case '2':
				if (eventName == "steam_2_end")
				{
					EndSteam2();
				}
				break;
			case '3':
				if (eventName == "steam_3_end")
				{
					EndSteam3();
				}
				break;
			case '5':
				if (eventName == "steam_5_end")
				{
					EndSteam5();
				}
				break;
			case '4':
				break;
			}
			break;
		case 8:
			switch (eventName[7])
			{
			case '1':
				if (eventName == "buildup1")
				{
					Buildup1();
				}
				break;
			case '2':
				if (eventName == "buildup2")
				{
					Buildup2();
				}
				break;
			case '3':
				if (eventName == "buildup3")
				{
					Buildup3();
				}
				break;
			}
			break;
		case 15:
			if (eventName == "waterfall_start")
			{
				StartWaterfall();
			}
			break;
		case 7:
			if (eventName == "explode")
			{
				Explode();
			}
			break;
		case 9:
		case 10:
		case 12:
		case 14:
			break;
		}
	}

	private void StartSteam1()
	{
		EmitGracefully(_steam1Particles);
	}

	private void EndSteam1()
	{
		_steam1Particles.Emitting = false;
	}

	private void StartSteam2()
	{
		EmitGracefully(_steam2Particles);
	}

	private void EndSteam2()
	{
		_steam2Particles.Emitting = false;
	}

	private void StartSteam3()
	{
		EmitGracefully(_steam3Particles);
		EmitGracefully(_steam4Particles);
	}

	private void EndSteam3()
	{
		_steam3Particles.Emitting = false;
		_steam4Particles.Emitting = false;
	}

	private void StartSteam5()
	{
		EmitGracefully(_steam5Particles);
		EmitGracefully(_steam6Particles);
	}

	private void EndSteam5()
	{
		_steam5Particles.Emitting = false;
		_steam6Particles.Emitting = false;
	}

	private void StartWaterfall()
	{
		_mouthParticles.Emitting = true;
		_dropletParticles.Emitting = true;
		_mistParticles.Emitting = true;
		_isDead = false;
	}

	private void EndWaterfall()
	{
		_mouthParticles.Emitting = false;
		_dropletParticles.Emitting = false;
		_mistParticles.Emitting = false;
	}

	private void Explode()
	{
		_steam1Particles.Visible = false;
		_steam2Particles.Visible = false;
		_steam3Particles.Visible = false;
		_steam4Particles.Visible = false;
		_steam5Particles.Visible = false;
		_steam6Particles.Visible = false;
		_steamLeakParticles1.Visible = false;
		_steamLeakParticles2.Visible = false;
		_steamLeakParticles3.Visible = false;
		_isDead = true;
	}

	private void Buildup1()
	{
		if (!_isDead)
		{
			EmitGracefully(_steamLeakParticles1);
			EmitGracefully(_steamLeakParticles2);
			EmitGracefully(_steamLeakParticles3);
			GpuParticles2D steamLeakParticles = _steamLeakParticles1;
			GpuParticles2D steamLeakParticles2 = _steamLeakParticles2;
			int num = (_steamLeakParticles3.Amount = 8);
			int amount = (steamLeakParticles2.Amount = num);
			steamLeakParticles.Amount = amount;
			GpuParticles2D steamLeakParticles3 = _steamLeakParticles1;
			GpuParticles2D steamLeakParticles4 = _steamLeakParticles2;
			double num4 = (_steamLeakParticles3.Lifetime = 0.3700000047683716);
			double lifetime = (steamLeakParticles4.Lifetime = num4);
			steamLeakParticles3.Lifetime = lifetime;
		}
	}

	private void Buildup2()
	{
		if (!_isDead)
		{
			EmitGracefully(_steamLeakParticles1);
			EmitGracefully(_steamLeakParticles2);
			EmitGracefully(_steamLeakParticles3);
			GpuParticles2D steamLeakParticles = _steamLeakParticles1;
			GpuParticles2D steamLeakParticles2 = _steamLeakParticles2;
			int num = (_steamLeakParticles3.Amount = 15);
			int amount = (steamLeakParticles2.Amount = num);
			steamLeakParticles.Amount = amount;
			GpuParticles2D steamLeakParticles3 = _steamLeakParticles1;
			GpuParticles2D steamLeakParticles4 = _steamLeakParticles2;
			double num4 = (_steamLeakParticles3.Lifetime = 0.44999998807907104);
			double lifetime = (steamLeakParticles4.Lifetime = num4);
			steamLeakParticles3.Lifetime = lifetime;
		}
	}

	private void Buildup3()
	{
		if (!_isDead)
		{
			EmitGracefully(_steamLeakParticles1);
			EmitGracefully(_steamLeakParticles2);
			EmitGracefully(_steamLeakParticles3);
			GpuParticles2D steamLeakParticles = _steamLeakParticles1;
			GpuParticles2D steamLeakParticles2 = _steamLeakParticles2;
			int num = (_steamLeakParticles3.Amount = 20);
			int amount = (steamLeakParticles2.Amount = num);
			steamLeakParticles.Amount = amount;
			GpuParticles2D steamLeakParticles3 = _steamLeakParticles1;
			GpuParticles2D steamLeakParticles4 = _steamLeakParticles2;
			double num4 = (_steamLeakParticles3.Lifetime = 0.6000000238418579);
			double lifetime = (steamLeakParticles4.Lifetime = num4);
			steamLeakParticles3.Lifetime = lifetime;
		}
	}

	private void EmitGracefully(GpuParticles2D emitter)
	{
		if (!emitter.Visible)
		{
			emitter.Visible = true;
			emitter.Restart();
		}
		else
		{
			emitter.Emitting = true;
		}
	}
}
