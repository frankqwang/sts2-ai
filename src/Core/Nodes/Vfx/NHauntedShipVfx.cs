using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NHauntedShipVfx : Node
{
	private MegaSprite _megaSprite;

	private GpuParticles2D _eyeParticles1;

	private GpuParticles2D _eyeParticles2;

	private GpuParticles2D _eyeParticles3;

	private GpuParticles2D _headParticles1;

	private GpuParticles2D _headParticles2;

	public override void _Ready()
	{
		_eyeParticles1 = GetNode<GpuParticles2D>("../EyeBone1/BubbleParticles");
		_eyeParticles2 = GetNode<GpuParticles2D>("../EyeBone2/BubbleParticles");
		_eyeParticles3 = GetNode<GpuParticles2D>("../EyeBone3/BubbleParticles");
		_headParticles1 = GetNode<GpuParticles2D>("../HeadSlot/BubbleParticles");
		_headParticles2 = GetNode<GpuParticles2D>("../HeadSlot/BubbleParticles2");
		_eyeParticles1.Emitting = false;
		_eyeParticles2.Emitting = false;
		_eyeParticles3.Emitting = false;
		_megaSprite = new MegaSprite(GetParent<Node2D>());
		_megaSprite.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		switch (new MegaEvent(spineEvent).GetData().GetEventName())
		{
		case "eye_bubbles_start":
			OnEyeBubblesStart();
			break;
		case "eye_bubbles_end":
			OnEyeBubblesEnd();
			break;
		case "head_bubbles_start":
			OnHeadBubblesStart();
			break;
		case "head_bubbles_end":
			OnHeadBubblesEnd();
			break;
		}
	}

	private void OnEyeBubblesStart()
	{
		_eyeParticles1.Emitting = true;
		_eyeParticles2.Emitting = true;
		_eyeParticles3.Emitting = true;
	}

	private void OnEyeBubblesEnd()
	{
		_eyeParticles1.Emitting = false;
		_eyeParticles2.Emitting = false;
		_eyeParticles3.Emitting = false;
	}

	private void OnHeadBubblesStart()
	{
		_headParticles1.Emitting = true;
		_headParticles2.Emitting = true;
	}

	private void OnHeadBubblesEnd()
	{
		_headParticles1.Emitting = false;
		_headParticles2.Emitting = false;
	}
}
