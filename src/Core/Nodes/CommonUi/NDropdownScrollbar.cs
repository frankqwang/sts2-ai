using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.CommonUi;

public partial class NDropdownScrollbar : NButton
{
	private NDropdownContainer _dropdownContainer;

	private Control _train;

	public bool hasControl;

	private Vector2 _startDragPos;

	private Vector2 _targetDragPos;

	private float _scrollLimitTop;

	private float _scrollLimitBottom;

	public override void _Ready()
	{
		ConnectSignals();
		_dropdownContainer = GetParent<NDropdownContainer>();
		_train = GetNode<Control>("Train");
		Connect(CanvasItem.SignalName.VisibilityChanged, Callable.From(OnShow));
	}

	public void RefreshTrainBounds()
	{
		_scrollLimitTop = 600f - _train.Size.Y - 9f;
		_scrollLimitBottom = 9f;
	}

	protected override void OnFocus()
	{
		_train.Modulate = StsColors.gold;
	}

	protected override void OnUnfocus()
	{
		if (!hasControl)
		{
			_train.Modulate = StsColors.quarterTransparentWhite;
		}
	}

	private void OnShow()
	{
		_train.Modulate = StsColors.quarterTransparentWhite;
	}

	protected override void OnPress()
	{
		hasControl = true;
		_train.Modulate = StsColors.gold;
		Input.MouseMode = Input.MouseModeEnum.Hidden;
	}

	public override void _Input(InputEvent inputEvent)
	{
		base._Input(inputEvent);
		if (hasControl && inputEvent is InputEventMouseButton inputEventMouseButton && inputEventMouseButton.ButtonIndex == MouseButton.Left && inputEventMouseButton.IsReleased())
		{
			hasControl = false;
			Input.MouseMode = Input.MouseModeEnum.Visible;
			_train.Modulate = StsColors.quarterTransparentWhite;
		}
	}

	public override void _GuiInput(InputEvent inputEvent)
	{
		base._GuiInput(inputEvent);
		if (hasControl && inputEvent is InputEventMouseMotion inputEventMouseMotion)
		{
			_train.Position += new Vector2(0f, inputEventMouseMotion.Relative.Y);
			ClampTrain();
			_dropdownContainer.UpdatePositionBasedOnTrain(1f - (_train.Position.Y - _scrollLimitBottom) / (_scrollLimitTop - _scrollLimitBottom));
		}
	}

	private void ClampTrain()
	{
		_train.Position = new Vector2(_train.Position.X, Mathf.Clamp(_train.Position.Y, _scrollLimitBottom, _scrollLimitTop));
	}

	public void SetTrainPositionFromPercentage(float percentage)
	{
		_train.Position = new Vector2(_train.Position.X, _scrollLimitBottom + percentage * (_scrollLimitTop - _scrollLimitBottom));
	}
}
