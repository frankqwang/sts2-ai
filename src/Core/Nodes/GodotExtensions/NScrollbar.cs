using Godot;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.Nodes.GodotExtensions;

[GlobalClass]
public partial class NScrollbar : Range
{
	[Signal]
	public delegate void MouseReleasedEventHandler(InputEvent inputEvent);

	[Signal]
	public delegate void MousePressedEventHandler(InputEvent inputEvent);

	private Control _handle;

	private float _currentHandlePosition;

	private float _currentVelocity;

	private bool _isDragging;

	public override void _Ready()
	{
		_handle = GetNode<Control>("%Handle");
	}

	public override void _GuiInput(InputEvent inputEvent)
	{
		base._GuiInput(inputEvent);
		if (!(inputEvent is InputEventMouseButton { ButtonIndex: var buttonIndex } inputEventMouseButton))
		{
			if (inputEvent is InputEventMouseMotion inputEventMouseMotion && _isDragging)
			{
				SetValueBasedOnMousePosition(inputEventMouseMotion.Position);
			}
		}
		else if (((ulong)(buttonIndex - 1) <= 1uL) ? true : false)
		{
			_isDragging = inputEventMouseButton.IsPressed();
			SetValueBasedOnMousePosition(inputEventMouseButton.Position);
			EmitSignal(inputEventMouseButton.IsPressed() ? SignalName.MousePressed : SignalName.MouseReleased, inputEvent);
		}
	}

	private void SetValueBasedOnMousePosition(Vector2 mousePosition)
	{
		base.Value = (double)(mousePosition.Y / base.Size.Y) * base.MaxValue;
	}

	public void SetValueWithoutAnimation(double value)
	{
		_currentHandlePosition = (float)value;
		base.Value = value;
		UpdateHandlePosition();
	}

	public override void _Process(double delta)
	{
		_currentHandlePosition = (float)base.Value;
		UpdateHandlePosition();
	}

	private void UpdateHandlePosition()
	{
		float y = MathHelper.SmoothDamp(_handle.Position.Y, base.Size.Y * (float)((double)_currentHandlePosition / base.MaxValue) - _handle.Size.Y * 0.5f, ref _currentVelocity, 0.05f, (float)GetProcessDeltaTime());
		_handle.Position = new Vector2((base.Size.X - _handle.Size.X) * 0.5f, y);
	}
}
