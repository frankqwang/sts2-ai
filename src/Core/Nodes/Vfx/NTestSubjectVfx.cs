using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NTestSubjectVfx : Node
{
	private GpuParticles2D _neckParticles;

	private GpuParticles2D _dizzyParticles;

	private GpuParticles2D _emberParticles;

	private GpuParticles2D _flameParticles;

	private GpuParticles2D _burnParticles;

	private GpuParticles2D _targetedBurnParticle;

	private GpuParticles2D _burnParticleFountain;

	private GpuParticles2D _ceilingParticles;

	private Node2D _parent;

	private MegaSprite _animController;

	private MegaSprite _frontBurnVfxController;

	private MegaSprite _backBurnVfxController;

	private bool _keyDown;

	private bool _doingThing;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_frontBurnVfxController = new MegaSprite(GetNode("../FrontBurnVfxSlot/FrontBurnVfx"));
		_backBurnVfxController = new MegaSprite(GetNode("../BackBurnVfxSlot/BackBurnVfx"));
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_neckParticles = _parent.GetNode<GpuParticles2D>("NeckParticlesSlot/NeckParticles");
		_dizzyParticles = _parent.GetNode<GpuParticles2D>("NeckParticlesSlot/DizzyPaticles");
		_emberParticles = _parent.GetNode<GpuParticles2D>("../../EmberParticles");
		_flameParticles = _parent.GetNode<GpuParticles2D>("../../FlameParticles");
		_burnParticles = _parent.GetNode<GpuParticles2D>("../../BurnParticles");
		_targetedBurnParticle = _parent.GetNode<GpuParticles2D>("../../TargetedBurnParticle");
		_burnParticleFountain = _parent.GetNode<GpuParticles2D>("../../BurnParticleFountain");
		_ceilingParticles = _parent.GetNode<GpuParticles2D>("../../CeilingSparks");
		_neckParticles.OneShot = true;
		_neckParticles.Emitting = false;
		_dizzyParticles.Emitting = false;
		_emberParticles.OneShot = true;
		_emberParticles.Emitting = false;
		_flameParticles.Emitting = false;
		_burnParticles.Emitting = false;
		_targetedBurnParticle.Emitting = false;
		_burnParticleFountain.Emitting = false;
		_ceilingParticles.OneShot = true;
		_ceilingParticles.Emitting = false;
		_animController.GetAnimationState().SetAnimation("idle_loop3");
		_frontBurnVfxController.GetAnimationState().SetAnimation("empty");
		_backBurnVfxController.GetAnimationState().SetAnimation("empty");
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
		case 12:
			switch (eventName[6])
			{
			case 'x':
				if (eventName == "neck_explode")
				{
					SquirtNeck();
				}
				break;
			case 'e':
				if (eventName == "start_embers")
				{
					StartEmbers();
				}
				break;
			case 'f':
				if (eventName == "start_flames")
				{
					StartFlames();
				}
				break;
			case 'r':
				if (eventName == "end_burn_vfx")
				{
					EndBurnVfx();
				}
				break;
			}
			break;
		case 13:
			if (eventName == "start_dizzies")
			{
				StartDizzies();
			}
			break;
		case 11:
			if (eventName == "end_dizzies")
			{
				EndDizzies();
			}
			break;
		case 10:
			if (eventName == "end_flames")
			{
				EndFlames();
			}
			break;
		case 14:
			if (eventName == "start_burn_vfx")
			{
				StartBurnVfx();
			}
			break;
		case 20:
			if (eventName == "start_ceiling_sparks")
			{
				StartCeilingSparks();
			}
			break;
		case 15:
		case 16:
		case 17:
		case 18:
		case 19:
			break;
		}
	}

	private void PlayAnim1()
	{
		_animController.GetAnimationState().SetAnimation("die3", loop: false);
		_animController.GetAnimationState().AddAnimation("idle_loop3");
	}

	private void SquirtNeck()
	{
		_neckParticles.Restart();
	}

	private void StartDizzies()
	{
		if (!_dizzyParticles.Emitting)
		{
			_dizzyParticles.Emitting = true;
		}
	}

	private void EndDizzies()
	{
		_dizzyParticles.Emitting = false;
	}

	private void StartEmbers()
	{
		_emberParticles.Restart();
	}

	private void StartFlames()
	{
		_flameParticles.Emitting = true;
	}

	private void EndFlames()
	{
		_flameParticles.Emitting = false;
	}

	private void StartBurnVfx()
	{
		_frontBurnVfxController.GetAnimationState().SetAnimation("burn", loop: false);
		_backBurnVfxController.GetAnimationState().SetAnimation("burn", loop: false);
		_burnParticles.Restart();
		_targetedBurnParticle.Emitting = true;
		_burnParticleFountain.Restart();
	}

	private void EndBurnVfx()
	{
		_burnParticles.Emitting = false;
		_targetedBurnParticle.Emitting = false;
		_burnParticleFountain.Emitting = false;
	}

	private void StartCeilingSparks()
	{
		_ceilingParticles.Restart();
	}
}
