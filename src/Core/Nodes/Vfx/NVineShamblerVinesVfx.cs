using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Rooms;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NVineShamblerVinesVfx : Node2D
{
	private Node2D _frontVinesNode;

	private MegaSprite _frontVinesAnimController;

	private Node2D _backVinesNode;

	private MegaSprite _backVinesAnimController;

	private GpuParticles2D _dirtBlast1;

	private GpuParticles2D _dirtBlast2;

	private GpuParticles2D _dirtBlast3;

	private GpuParticles2D _dirtBlast4;

	public override void _Ready()
	{
		_frontVinesNode = GetNode<Node2D>("VinesFront");
		_frontVinesAnimController = new MegaSprite(_frontVinesNode);
		_frontVinesAnimController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnFrontEvent));
		_frontVinesAnimController.ConnectAnimationCompleted(Callable.From<GodotObject, GodotObject, GodotObject>(AnimationEnded));
		_backVinesNode = GetNode<Node2D>("VinesBackScene/VinesBack");
		_backVinesAnimController = new MegaSprite(_backVinesNode);
		_dirtBlast1 = GetNode<GpuParticles2D>("DirtBlast1");
		_dirtBlast3 = GetNode<GpuParticles2D>("DirtBlast3");
		_dirtBlast2 = GetNode<GpuParticles2D>("VinesBackScene/DirtBlast2");
		_dirtBlast4 = GetNode<GpuParticles2D>("VinesBackScene/DirtBlast4");
		_dirtBlast1.Emitting = false;
		_dirtBlast1.OneShot = true;
		_dirtBlast2.Emitting = false;
		_dirtBlast2.OneShot = true;
		_dirtBlast3.Emitting = false;
		_dirtBlast3.OneShot = true;
		_dirtBlast4.Emitting = false;
		_dirtBlast4.OneShot = true;
		Vector2 backVineOffset = _backVinesNode.GlobalPosition - _frontVinesNode.GlobalPosition;
		_backVinesNode.Reparent(NCombatRoom.Instance.BackCombatVfxContainer);
		Callable.From(delegate
		{
			_backVinesNode.GlobalPosition = _frontVinesNode.GlobalPosition + backVineOffset;
		}).CallDeferred();
		_frontVinesAnimController.GetAnimationState().SetAnimation("animation");
		_backVinesAnimController.GetAnimationState().SetAnimation("animation");
	}

	private void OnFrontEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		switch (new MegaEvent(spineEvent).GetData().GetEventName())
		{
		case "dirt_1":
			_dirtBlast1.Restart();
			break;
		case "dirt_2":
			_dirtBlast2.Restart();
			break;
		case "dirt_3":
			_dirtBlast3.Restart();
			break;
		case "dirt_4":
			_dirtBlast4.Restart();
			break;
		}
	}

	private void AnimationEnded(GodotObject _, GodotObject __, GodotObject ___)
	{
		this.QueueFreeSafely();
		_backVinesNode.QueueFreeSafely();
	}
}
