using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Map;

public partial class NControllerMapDrawingInput : NMapDrawingInput
{
	private const string _scenePath = "res://scenes/screens/map/controller_map_drawing_input.tscn";

	private Vector2 _eraserIconPos = new Vector2(-34f, -76f);

	private Vector2 _drawingIconPos = new Vector2(-10f, -76f);

	private bool _isPressed;

	private Texture2D _cursorTex;

	private Texture2D _cursorTiltedTex;

	private Control _cursor;

	private Vector2 _direction;

	public static NMapDrawingInput Create()
	{
		return PreloadManager.Cache.GetScene("res://scenes/screens/map/controller_map_drawing_input.tscn").Instantiate<NControllerMapDrawingInput>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		base._Ready();
		_cursor = GetNode<Control>("%Cursor");
		this.TryGrabFocus();
		GetViewport().Connect(Viewport.SignalName.GuiFocusChanged, Callable.From<Control>(delegate
		{
			StopDrawing();
		}));
		if (base.DrawingMode == DrawingMode.Drawing)
		{
			_cursorTex = ImageTexture.CreateFromImage(PreloadManager.Cache.GetAsset<Image>("res://images/packed/common_ui/cursor_quill.png"));
			_cursorTiltedTex = ImageTexture.CreateFromImage(PreloadManager.Cache.GetAsset<Image>("res://images/packed/common_ui/cursor_quill_tilted.png"));
		}
		else
		{
			_cursorTex = ImageTexture.CreateFromImage(PreloadManager.Cache.GetAsset<Image>("res://images/packed/common_ui/cursor_eraser.png"));
			_cursorTiltedTex = ImageTexture.CreateFromImage(PreloadManager.Cache.GetAsset<Image>("res://images/packed/common_ui/cursor_eraser_tilted.png"));
		}
		_cursor.GetNode<TextureRect>("TextureRect").Texture = _cursorTex;
		_cursor.GetNode<TextureRect>("TextureRect").Position = ((base.DrawingMode == DrawingMode.Drawing) ? _drawingIconPos : _eraserIconPos);
	}

	public override void _Process(double delta)
	{
		if (Input.IsActionPressed(MegaInput.select))
		{
			if (!_isPressed)
			{
				_drawings.BeginLineLocal(_drawings.GetGlobalTransform().Inverse() * _cursor.GlobalPosition, null);
				_cursor.GetNode<TextureRect>("TextureRect").Texture = _cursorTiltedTex;
				_isPressed = true;
			}
		}
		else if (_isPressed)
		{
			_drawings.StopLineLocal();
			_cursor.GetNode<TextureRect>("TextureRect").Texture = _cursorTex;
			_isPressed = false;
		}
		_direction = Input.GetVector(Controller.joystickLeft, Controller.joystickRight, Controller.joystickUp, Controller.joystickDown);
		if (_direction.Length() < 0.1f)
		{
			_direction += Input.GetVector(Controller.dPadWest, Controller.dPadEast, Controller.dPadNorth, Controller.dPadSouth);
		}
		if (_direction.Length() > 0f)
		{
			_cursor.GlobalPosition += _direction * 700f * (float)delta;
			_cursor.GlobalPosition = _cursor.GlobalPosition.Clamp(NGame.Instance.GlobalPosition, NGame.Instance.GlobalPosition + NGame.Instance.Size);
			if (_drawings.IsLocalDrawing())
			{
				_drawings.UpdateCurrentLinePositionLocal(_drawings.GetGlobalTransform().Inverse() * _cursor.GlobalPosition);
			}
		}
	}

	public override void _Input(InputEvent input)
	{
		if (IsVisibleInTree() && input.IsActionPressed(MegaInput.cancel))
		{
			StopDrawing();
			ActiveScreenContext.Instance.FocusOnDefaultControl();
		}
	}
}
