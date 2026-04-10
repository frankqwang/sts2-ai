using Godot;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Map;

public partial class NMouseHeldMapDrawingInput : NMapDrawingInput
{
	private MouseButton ListeningButton
	{
		get
		{
			if (base.DrawingMode != DrawingMode.Drawing)
			{
				return MouseButton.Middle;
			}
			return MouseButton.Right;
		}
	}

	public override void _Ready()
	{
		base._Ready();
		_drawings.BeginLineLocal(_drawings.GetGlobalTransform().Inverse() * GetGlobalMousePosition(), null);
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (IsVisibleInTree())
		{
			ProcessMouseDrawingEvent(inputEvent);
			if (inputEvent is InputEventMouseMotion inputEventMouseMotion && _drawings.IsLocalDrawing())
			{
				_drawings.UpdateCurrentLinePositionLocal(_drawings.GetGlobalTransform().Inverse() * inputEventMouseMotion.GlobalPosition);
			}
		}
	}

	private void ProcessMouseDrawingEvent(InputEvent inputEvent)
	{
		if (inputEvent is InputEventMouseButton inputEventMouseButton && inputEventMouseButton.ButtonIndex == ListeningButton && !inputEventMouseButton.IsPressed())
		{
			StopDrawing();
		}
	}
}
