using Godot;

namespace MegaCrit.Sts2.Core.Nodes.CommonUi;

public partial class NScrollbarTrain : TextureRect
{
	private Tween? _tween;

	public override void _Ready()
	{
		Connect(Control.SignalName.MouseEntered, Callable.From(OnMouseEntered));
		Connect(Control.SignalName.MouseExited, Callable.From(OnMouseExited));
	}

	private void OnMouseEntered()
	{
		_tween?.Kill();
		base.Scale = Vector2.One * 1.15f;
	}

	private void OnMouseExited()
	{
		_tween?.Kill();
		_tween = CreateTween();
		_tween.TweenProperty(this, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}
}
