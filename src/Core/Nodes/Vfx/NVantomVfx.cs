using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NVantomVfx : Node
{
	private static readonly StringName _step = new StringName("step");

	private ShaderMaterial? _tailShaderMat;

	private GpuParticles2D _sprayParticles;

	private GpuParticles2D _deathSprayParticles;

	private Node2D _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_tailShaderMat = new MegaSlotNode(_parent.GetNode("TailSlotNode")).GetNormalMaterial() as ShaderMaterial;
		_sprayParticles = _parent.GetNode<GpuParticles2D>("SprayBoneNode/SprayParticles");
		_deathSprayParticles = _parent.GetNode<GpuParticles2D>("DeathSpraySlotNode/DeathSprayParticles");
		_tailShaderMat?.SetShaderParameter(_step, -0.1f);
		_animController.GetAnimationState().SetAnimation("idle_loop");
		_animController.GetAnimationState().SetAnimation("_tracks/charged_0", loop: true, 1);
		_sprayParticles.Emitting = false;
		_deathSprayParticles.Emitting = false;
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		switch (new MegaEvent(spineEvent).GetData().GetEventName())
		{
		case "dissolve_tail":
			DissolveTail();
			break;
		case "spray_on":
			StartSpray();
			break;
		case "spray_off":
			EndSpray();
			break;
		case "death_spray_on":
			StartDeathSpray();
			break;
		case "death_spray_off":
			EndDeathSpray();
			break;
		}
	}

	private void DissolveTail()
	{
		if (_tailShaderMat != null)
		{
			Tween tween = CreateTween();
			tween.SetEase(Tween.EaseType.In);
			tween.SetTrans(Tween.TransitionType.Quad);
			tween.TweenProperty(_tailShaderMat, "shader_parameter/step", 1f, 1.0);
			tween.TweenCallback(Callable.From(delegate
			{
				_animController.GetAnimationState().SetAnimation("_tracks/charge_up_1", loop: false, 1);
				_animController.GetAnimationState().AddAnimation("_tracks/charged_1", 0f, loop: true, 1);
			}));
		}
	}

	private void StartSpray()
	{
		_sprayParticles.Emitting = true;
	}

	private void EndSpray()
	{
		_sprayParticles.Emitting = false;
	}

	private void StartDeathSpray()
	{
		_deathSprayParticles.Emitting = true;
	}

	private void EndDeathSpray()
	{
		_deathSprayParticles.Emitting = false;
	}
}
