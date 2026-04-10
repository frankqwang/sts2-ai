using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Timeline;

public partial class NResetProgressButton : Control
{
	private MegaLabel _disclaimer;

	public override void _Ready()
	{
		Connect(Control.SignalName.MouseEntered, Callable.From(OnMouseEntered));
		Connect(Control.SignalName.MouseExited, Callable.From(OnMouseExited));
		Connect(Control.SignalName.GuiInput, Callable.From<InputEvent>(OnGuiInput));
		_disclaimer = GetNode<MegaLabel>("%CloseScreenDisclaimer");
	}

	private void OnMouseExited()
	{
		base.Scale = Vector2.One;
	}

	private void OnMouseEntered()
	{
		base.Scale = Vector2.One * 1.1f;
	}

	private void OnGuiInput(InputEvent inputEvent)
	{
		if (inputEvent is InputEventMouseButton inputEventMouseButton && inputEventMouseButton.ButtonIndex == MouseButton.Left && !inputEventMouseButton.Pressed)
		{
			SaveManager.Instance.ResetTimelineProgress();
			_disclaimer.Visible = true;
		}
	}
}
