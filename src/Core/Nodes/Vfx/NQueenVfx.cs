using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NQueenVfx : Node2D
{
	[Export(PropertyHint.None, "")]
	private Node2D _spineNode;

	[Export(PropertyHint.None, "")]
	private GpuParticles2D _sprayParticles;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_animController = new MegaSprite(_spineNode);
		_animController.ConnectAnimationStarted(Callable.From<GodotObject, GodotObject, GodotObject>(OnAnimationStart));
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_sprayParticles.Emitting = false;
	}

	private void OnAnimationEvent(GodotObject sprite, GodotObject animationState, GodotObject trackEntry, GodotObject eventObject)
	{
		string eventName = new MegaEvent(eventObject).GetData().GetEventName();
		if (!(eventName == "attack_start"))
		{
			if (eventName == "attack_end")
			{
				EndAttack();
			}
		}
		else
		{
			StartAttack();
		}
	}

	private void OnAnimationStart(GodotObject spineSprite, GodotObject animationState, GodotObject trackEntry)
	{
		if (new MegaAnimationState(animationState).GetCurrent(0).GetAnimation().GetName() != "attack")
		{
			EndAttack();
		}
	}

	private void StartAttack()
	{
		_sprayParticles.Restart();
	}

	private void EndAttack()
	{
		_sprayParticles.Emitting = false;
	}
}
