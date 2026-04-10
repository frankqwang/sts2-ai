using Godot;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.StatsScreen;

public partial class NShareStatsButton : NButton
{
	private Control _image;

	public override void _Ready()
	{
		ConnectSignals();
		_image = GetNode<Control>("Visuals");
	}

	protected override void OnRelease()
	{
		_image.Scale = Vector2.One;
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_image.Scale = Vector2.One * 1.05f;
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		_image.Scale = Vector2.One;
	}

	protected override void OnPress()
	{
		base.OnPress();
		_image.Scale = Vector2.One * 0.95f;
	}
}
