using Godot;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.CardLibrary;

public partial class NClearSearchButton : NButton
{
	private Tween? _tween;

	private Control _image;

	public override void _Ready()
	{
		ConnectSignals();
		_image = GetNode<Control>("Image");
		_image.PivotOffset = _image.Size / 2f;
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_image, "scale", Vector2.One * 1.1f, 0.05);
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_image, "scale", Vector2.One, 0.3);
	}
}
