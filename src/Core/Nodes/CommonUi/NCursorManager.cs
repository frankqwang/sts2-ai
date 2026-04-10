using Godot;

namespace MegaCrit.Sts2.Core.Nodes.CommonUi;

public partial class NCursorManager : Node
{
	private static readonly Vector2 _defaultHotSpot = new Vector2(14f, 5f);

	private static readonly Vector2 _inspectHotSpot = new Vector2(12f, 12f);

	[Export(PropertyHint.None, "")]
	private Image _cursorTilted;

	[Export(PropertyHint.None, "")]
	private Image _cursorNotTilted;

	[Export(PropertyHint.None, "")]
	private Image _cursorInspect;

	private Image? _overriddenCursorTilted;

	private Image? _overriddenCursorNotTilted;

	private Vector2? _overriddenHotSpot;

	private Image? _lastSetCursor;

	private bool _isDown;

	private bool _isUsingController;

	private bool _shouldShowCursor = true;

	private Image CursorTilted => _overriddenCursorTilted ?? _cursorTilted;

	private Image CursorNotTilted => _overriddenCursorNotTilted ?? _cursorNotTilted;

	private Vector2 HotSpot => _overriddenHotSpot ?? _defaultHotSpot;

	public override void _EnterTree()
	{
		Input.SetCustomMouseCursor(_cursorInspect, Input.CursorShape.Help, _inspectHotSpot);
		UpdateCursor();
	}

	public override void _Ready()
	{
		NControllerManager.Instance.Connect(NControllerManager.SignalName.ControllerDetected, Callable.From(delegate
		{
			SetIsUsingController(isUsingController: true);
		}));
		NControllerManager.Instance.Connect(NControllerManager.SignalName.MouseDetected, Callable.From(delegate
		{
			SetIsUsingController(isUsingController: false);
		}));
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (inputEvent is InputEventMouseButton inputEventMouseButton && (inputEventMouseButton.ButtonIndex == MouseButton.Left || inputEventMouseButton.ButtonIndex == MouseButton.Right || inputEventMouseButton.ButtonIndex == MouseButton.Middle))
		{
			if (inputEventMouseButton.IsPressed() && !_isDown)
			{
				_isDown = true;
				UpdateCursor();
			}
			else if (inputEventMouseButton.IsReleased() && _isDown)
			{
				_isDown = false;
				UpdateCursor();
			}
		}
	}

	public void StopOverridingCursor()
	{
		_overriddenCursorTilted = null;
		_overriddenCursorNotTilted = null;
		_overriddenHotSpot = null;
		UpdateCursor();
	}

	public void OverrideCursor(Image cursorTilted, Image cursorNotTilted, Vector2 hotspot)
	{
		_overriddenCursorTilted = cursorTilted;
		_overriddenCursorNotTilted = cursorNotTilted;
		_overriddenHotSpot = hotspot;
		UpdateCursor();
	}

	private void UpdateCursor()
	{
		if (Input.MouseMode != Input.MouseModeEnum.Hidden)
		{
			Image image = (_isDown ? CursorTilted : CursorNotTilted);
			if (image != _lastSetCursor)
			{
				Input.SetCustomMouseCursor(image, Input.CursorShape.Arrow, HotSpot);
				_lastSetCursor = image;
			}
		}
	}

	private void SetIsUsingController(bool isUsingController)
	{
		_isUsingController = isUsingController;
		RefreshCursorShown();
	}

	public void SetCursorShown(bool show)
	{
		_shouldShowCursor = show;
		RefreshCursorShown();
	}

	private void RefreshCursorShown()
	{
		bool flag = !_isUsingController && _shouldShowCursor;
		Input.MouseMode = (Input.MouseModeEnum)(flag ? 0 : 1);
		if (!flag)
		{
			_lastSetCursor = null;
		}
	}
}
