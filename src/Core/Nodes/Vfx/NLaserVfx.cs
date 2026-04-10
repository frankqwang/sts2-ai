using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NLaserVfx : Node2D
{
	private static readonly StringName _color = new StringName("Color");

	private Node2D _animNode;

	private MegaSprite _animController;

	private Node2D _targetingBone;

	public override void _Ready()
	{
		_animNode = GetNode<Node2D>("SpineSprite");
		_animController = new MegaSprite(_animNode);
		_targetingBone = GetNode<Node2D>("SpineSprite/TargetingBone");
		_animController.GetAnimationState().SetAnimation("animation");
		_animNode.Visible = false;
	}

	public void ExtendLaser(Vector2 targetPos)
	{
		_animNode.Visible = true;
		_animController.GetAnimationState().SetAnimation("animation");
		_targetingBone.GlobalPosition = base.GlobalPosition;
		Tween tween = CreateTween();
		tween.TweenProperty(_targetingBone, "position", targetPos, 0.15000000596046448).SetTrans(Tween.TransitionType.Expo).SetEase(Tween.EaseType.Out);
		tween.Chain().TweenProperty(_animNode, "modulate", Colors.Red, 0.20000000298023224);
	}

	public void RetractLaser()
	{
		Tween tween = CreateTween();
		tween.TweenProperty(_targetingBone, "position", base.Position, 0.15000000596046448).SetTrans(Tween.TransitionType.Expo).SetEase(Tween.EaseType.In);
		tween.Chain().TweenProperty(_animNode, "visible", false, 0.0);
	}

	public void ResetLaser()
	{
		_targetingBone.Position = base.Position;
	}

	private void SetLaserColor(Color color)
	{
		((ShaderMaterial)_animController.GetAdditiveMaterial()).SetShaderParameter(_color, color);
	}
}
