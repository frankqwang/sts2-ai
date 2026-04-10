using Godot;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.Nodes.CommonUi;

public partial class NDropdownContainer : Control
{
	private NDropdownScrollbar _scrollbar;

	private Control _scrollbarTrain;

	private VBoxContainer _dropdownItems;

	private float _maxHeight;

	private float _contentHeight;

	private Vector2 _startDragPos;

	private Vector2 _targetDragPos = Vector2.Zero;

	private const float _scrollLimitTop = 0f;

	private float _scrollLimitBottom;

	private bool _isDragging;

	public override void _Ready()
	{
		_scrollbar = GetNode<NDropdownScrollbar>("Scrollbar");
		_scrollbarTrain = GetNode<Control>("Scrollbar/Train");
		_dropdownItems = GetNode<VBoxContainer>("VBoxContainer");
		Connect(CanvasItem.SignalName.VisibilityChanged, Callable.From(OnVisibilityChange));
		_maxHeight = base.Size.Y;
	}

	public override void _EnterTree()
	{
		base._EnterTree();
		GetViewport().Connect(Viewport.SignalName.GuiFocusChanged, Callable.From<Control>(ProcessGuiFocus));
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		GetViewport().Disconnect(Viewport.SignalName.GuiFocusChanged, Callable.From<Control>(ProcessGuiFocus));
	}

	private void OnVisibilityChange()
	{
		if (base.Visible)
		{
			_isDragging = false;
		}
	}

	public void RefreshLayout()
	{
		_scrollbar.Visible = IsScrollbarNeeded();
	}

	private bool IsScrollbarNeeded()
	{
		_contentHeight = 0f;
		foreach (Node child in _dropdownItems.GetChildren())
		{
			if (child is Control control)
			{
				_contentHeight += control.Size.Y;
			}
		}
		_scrollLimitBottom = 0f - _contentHeight + _maxHeight;
		if (_contentHeight > _maxHeight)
		{
			base.Size = new Vector2(base.Size.X, _maxHeight);
			_scrollbar.RefreshTrainBounds();
			return true;
		}
		base.Size = new Vector2(base.Size.X, _contentHeight);
		return false;
	}

	public override void _Process(double delta)
	{
		if (IsVisibleInTree())
		{
			UpdateScrollPosition(delta);
			UpdateScrollbar();
		}
	}

	private void ProcessGuiFocus(Control focusedControl)
	{
		if (IsVisibleInTree() && _scrollbar.Visible && NControllerManager.Instance.IsUsingController && _dropdownItems.IsAncestorOf(focusedControl))
		{
			float num = _dropdownItems.GlobalPosition.Y - focusedControl.GlobalPosition.Y;
			float value = num + base.Size.Y * 0.5f;
			value = Mathf.Clamp(value, _scrollLimitBottom, 0f);
			_targetDragPos = new Vector2(_targetDragPos.X, value);
		}
	}

	private void UpdateScrollPosition(double delta)
	{
		if (!_scrollbar.Visible)
		{
			return;
		}
		if (_dropdownItems.Position != _targetDragPos)
		{
			_dropdownItems.Position = _dropdownItems.Position.Lerp(_targetDragPos, (float)delta * 15f);
			if (_dropdownItems.Position.DistanceTo(_targetDragPos) < 0.5f)
			{
				_dropdownItems.Position = _targetDragPos;
			}
		}
		if (!_isDragging)
		{
			if (_targetDragPos.Y < _scrollLimitBottom)
			{
				_targetDragPos = _targetDragPos.Lerp(new Vector2(0f, _scrollLimitBottom), (float)delta * 12f);
			}
			else if (_targetDragPos.Y > 0f)
			{
				_targetDragPos = _targetDragPos.Lerp(new Vector2(0f, 0f), (float)delta * 12f);
			}
		}
	}

	private void UpdateScrollbar()
	{
		if (_scrollbar.Visible && !_scrollbar.hasControl)
		{
			float num = (_dropdownItems.Position.Y - _scrollLimitBottom) / (0f - _scrollLimitBottom);
			_scrollbar.SetTrainPositionFromPercentage(Mathf.Clamp(1f - num, 0f, 1f));
		}
	}

	public void UpdatePositionBasedOnTrain(float trainPosition)
	{
		_targetDragPos = new Vector2(_targetDragPos.X, _scrollLimitBottom + trainPosition * (0f - _scrollLimitBottom));
	}

	public override void _GuiInput(InputEvent inputEvent)
	{
		ProcessMouseEvent(inputEvent);
		ProcessScrollEvent(inputEvent);
	}

	private void ProcessMouseEvent(InputEvent inputEvent)
	{
		if (_isDragging && inputEvent is InputEventMouseMotion inputEventMouseMotion)
		{
			_targetDragPos += new Vector2(0f, inputEventMouseMotion.Relative.Y);
		}
		else
		{
			if (!(inputEvent is InputEventMouseButton inputEventMouseButton))
			{
				return;
			}
			if (inputEventMouseButton.ButtonIndex == MouseButton.Left)
			{
				if (inputEventMouseButton.Pressed)
				{
					_isDragging = true;
					_startDragPos = _dropdownItems.Position;
					_targetDragPos = _startDragPos;
				}
				else
				{
					_isDragging = false;
				}
			}
			else if (!inputEventMouseButton.Pressed)
			{
				_isDragging = false;
			}
		}
	}

	private void ProcessScrollEvent(InputEvent inputEvent)
	{
		_targetDragPos += new Vector2(0f, ScrollHelper.GetDragForScrollEvent(inputEvent));
	}
}
