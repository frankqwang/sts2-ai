using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;
using MegaCrit.Sts2.Core.Commands;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NKaiserCrabBossVfx : Node
{
	private MegaSprite _megaSprite;

	private GpuParticles2D _regenSplatParticles;

	private GpuParticles2D _plowChunkParticles;

	private GpuParticles2D _steamParticles1;

	private GpuParticles2D _steamParticles2;

	private GpuParticles2D _steamParticles3;

	private GpuParticles2D _smokeParticles;

	private GpuParticles2D _sparkParticles;

	private GpuParticles2D _spittleParticles;

	private Node2D _leftArmExplosionPosition;

	private Node2D _parent;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_regenSplatParticles = _parent.GetNode<GpuParticles2D>("RegenSplatSlot/RegenSplatParticles");
		_plowChunkParticles = _parent.GetNode<GpuParticles2D>("PlowChunkSlot/PlowChunkParticles");
		_steamParticles1 = _parent.GetNode<GpuParticles2D>("RocketSlot/SteamParticles1");
		_steamParticles2 = _parent.GetNode<GpuParticles2D>("RocketSlot/SteamParticles2");
		_steamParticles3 = _parent.GetNode<GpuParticles2D>("RocketSlot/SteamParticles3");
		_sparkParticles = _parent.GetNode<GpuParticles2D>("RocketSlot/SparkParticles");
		_smokeParticles = _parent.GetNode<GpuParticles2D>("RocketSlot/SmokeParticles");
		_leftArmExplosionPosition = _parent.GetNode<Node2D>("%LeftArmExplosionPosition");
		_spittleParticles = _parent.GetNode<GpuParticles2D>("SpittleSlot/SpittleParticles");
		_megaSprite = new MegaSprite(GetParent<Node2D>());
		_megaSprite.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_megaSprite.ConnectAnimationStarted(Callable.From<GodotObject, GodotObject, GodotObject>(OnAnimationStart));
		_regenSplatParticles.Emitting = false;
		_plowChunkParticles.Emitting = false;
		_steamParticles1.Emitting = false;
		_steamParticles2.Emitting = false;
		_steamParticles3.Emitting = false;
		_spittleParticles.Emitting = false;
		_sparkParticles.Emitting = false;
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
		case 18:
			switch (eventName[0])
			{
			case 'c':
				if (eventName == "charge_steam_start")
				{
					OnChargeSteamStart();
				}
				break;
			case 'r':
				if (eventName == "regen_splats_start")
				{
					OnRegenSplatsStart();
				}
				break;
			}
			break;
		case 16:
			switch (eventName[0])
			{
			case 'c':
				if (eventName == "charge_steam_end")
				{
					OnChargeSteamEnd();
				}
				break;
			case 'd':
				if (eventName == "death_spit_start")
				{
					OnDeathSpitStart();
				}
				break;
			case 'r':
				if (eventName == "regen_splats_end")
				{
					OnRegenSplatsEnd();
				}
				break;
			}
			break;
		case 14:
			switch (eventName[0])
			{
			case 'd':
				if (eventName == "death_spit_end")
				{
					OnDeathSpitEnd();
				}
				break;
			case 'c':
				if (eventName == "claw_explode_l")
				{
					OnClawLExplode();
				}
				break;
			}
			break;
		case 17:
			switch (eventName[0])
			{
			case 'p':
				if (eventName == "plow_chunks_start")
				{
					OnPlowChunksStart();
				}
				break;
			case 'r':
				if (eventName == "rocket_thrust_end")
				{
					OnRocketThrustEnd();
				}
				break;
			}
			break;
		case 15:
			if (eventName == "plow_chunks_end")
			{
				OnPlowChunksEnd();
			}
			break;
		case 19:
			if (eventName == "rocket_thrust_start")
			{
				OnRocketThrustStart();
			}
			break;
		}
	}

	private void OnAnimationStart(GodotObject spineSprite, GodotObject animationState, GodotObject trackEntry)
	{
		string name = new MegaAnimationState(animationState).GetCurrent(2).GetAnimation().GetName();
		if (name != "right/charged_loop" && name != "right/charge_up")
		{
			OnChargeSteamEnd();
		}
		if (name != "right/attack_heavy")
		{
			OnRocketThrustEnd();
		}
	}

	private void OnChargeSteamStart()
	{
		_steamParticles1.Restart();
		_steamParticles2.Restart();
		_steamParticles3.Restart();
	}

	private void OnChargeSteamEnd()
	{
		_steamParticles1.Emitting = false;
		_steamParticles2.Emitting = false;
		_steamParticles3.Emitting = false;
	}

	private void OnDeathSpitStart()
	{
		_spittleParticles.Restart();
	}

	private void OnDeathSpitEnd()
	{
		_spittleParticles.Emitting = false;
	}

	private void OnLeftEmbersStart()
	{
	}

	private void OnPlowChunksStart()
	{
		_plowChunkParticles.Restart();
	}

	private void OnPlowChunksEnd()
	{
		_plowChunkParticles.Emitting = false;
	}

	private void OnRegenSplatsStart()
	{
		_regenSplatParticles.Restart();
	}

	private void OnRegenSplatsEnd()
	{
		_regenSplatParticles.Emitting = false;
	}

	private void OnRocketThrustStart()
	{
		_smokeParticles.Restart();
		_sparkParticles.Restart();
	}

	private void OnRocketThrustEnd()
	{
		_smokeParticles.Emitting = false;
		_sparkParticles.Emitting = false;
	}

	private void OnClawLExplode()
	{
		VfxCmd.PlayVfx(_leftArmExplosionPosition.GlobalPosition, "vfx/monsters/kaiser_crab_boss_explosion");
	}
}
