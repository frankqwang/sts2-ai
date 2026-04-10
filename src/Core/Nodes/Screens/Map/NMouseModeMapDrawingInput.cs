using Godot;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Map;

public partial class NMouseModeMapDrawingInput : NMapDrawingInput
{
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
		if (!(inputEvent is InputEventMouseButton inputEventMouseButton))
		{
			return;
		}
		if (inputEventMouseButton.ButtonIndex == MouseButton.Left)
		{
			if (inputEventMouseButton.Pressed && !_drawings.IsLocalDrawing())
			{
				_drawings.BeginLineLocal(_drawings.GetGlobalTransform().Inverse() * inputEventMouseButton.GlobalPosition, null);
			}
			else if (!inputEventMouseButton.Pressed && _drawings.IsLocalDrawing())
			{
				_drawings.StopLineLocal();
			}
		}
		else
		{
			MouseButton buttonIndex = inputEventMouseButton.ButtonIndex;
			bool flag = (((ulong)(buttonIndex - 2) <= 1uL) ? true : false);
			if (flag && inputEventMouseButton.Pressed)
			{
				StopDrawing();
			}
		}
	}
}
