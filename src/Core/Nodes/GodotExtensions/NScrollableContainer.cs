using System;
using Godot;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.CommonUi;

namespace MegaCrit.Sts2.Core.Nodes.GodotExtensions;

public partial class NScrollableContainer : Control
{
	private float _controllerScrollAmount = 400f;

	private float _startDragPosY;

	private float _targetDragPosY;

	private bool _isDragging;

	private float _paddingTop;

	private float _paddingBottom;

	private Control? _content;

	private bool _scrollbarPressed;

	private bool _disableScrollingIfContentFits;

	private float ScrollViewportTop
	{
		get
		{
			if (_content != null)
			{
				return _content.GetParent<Control>().Position.Y;
			}
			return 0f;
		}
	}

	private float ScrollViewportSize
	{
		get
		{
			if (_content != null)
			{
				return _content.GetParent<Control>().Size.Y;
			}
			return 0f;
		}
	}

	private float ScrollLimitBottom
	{
		get
		{
			if (_content != null)
			{
				return 0f - (_paddingBottom + _paddingTop + _content.Size.Y) + _content.GetParent<Control>().Size.Y;
			}
			return 0f;
		}
	}

	public NScrollbar Scrollbar { get; private set; }

	public override void _Ready()
	{
		_content = GetNodeOrNull<Control>("Content") ?? GetNodeOrNull<Control>("Mask/Content");
		Scrollbar = GetNode<NScrollbar>("Scrollbar");
		SetContent(_content);
		Scrollbar.Visible = false;
		Scrollbar.Connect(NScrollbar.SignalName.MousePressed, Callable.From<InputEvent>(delegate
		{
			_scrollbarPressed = true;
		}));
		Scrollbar.Connect(NScrollbar.SignalName.MouseReleased, Callable.From<InputEvent>(delegate
		{
			_scrollbarPressed = false;
		}));
	}

	public void SetContent(Control? content, float paddingTop = 0f, float paddingBottom = 0f)
	{
		Callable callable = Callable.From(UpdateScrollLimitBottom);
		if (_content != null && _content.IsConnected(CanvasItem.SignalName.ItemRectChanged, callable))
		{
			_content.Disconnect(CanvasItem.SignalName.ItemRectChanged, callable);
		}
		_content = content;
		if (_content != null)
		{
			_content.Connect(CanvasItem.SignalName.ItemRectChanged, Callable.From(UpdateScrollLimitBottom));
			_paddingTop = paddingTop;
			_paddingBottom = paddingBottom;
			UpdateScrollLimitBottom();
		}
	}

	public void DisableScrollingIfContentFits()
	{
		_disableScrollingIfContentFits = true;
	}

	public override void _EnterTree()
	{
		GetViewport().Connect(Viewport.SignalName.GuiFocusChanged, Callable.From<Control>(ProcessGuiFocus));
	}

	public override void _ExitTree()
	{
		GetViewport().Disconnect(Viewport.SignalName.GuiFocusChanged, Callable.From<Control>(ProcessGuiFocus));
	}

	private void UpdateScrollLimitBottom()
	{
		if (_content != null)
		{
			Scrollbar.Visible = _content.Size.Y + _paddingTop + _paddingBottom > ScrollViewportSize;
			Scrollbar.MouseFilter = (MouseFilterEnum)(Scrollbar.Visible ? 0 : 2);
		}
	}

	public override void _GuiInput(InputEvent inputEvent)
	{
		if (IsVisibleInTree())
		{
			ProcessMouseEvent(inputEvent);
			ProcessScrollEvent(inputEvent);
		}
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (IsVisibleInTree())
		{
			Viewport viewport = GetViewport();
			if (viewport == null || viewport.GuiGetFocusOwner() == null)
			{
				ProcessControllerEvent(inputEvent);
			}
		}
	}

	private void ProcessControllerEvent(InputEvent inputEvent)
	{
		if (inputEvent.IsActionPressed(MegaInput.up))
		{
			_targetDragPosY += _controllerScrollAmount;
		}
		else if (inputEvent.IsActionPressed(MegaInput.down))
		{
			_targetDragPosY += 0f - _controllerScrollAmount;
		}
	}

	private void ProcessMouseEvent(InputEvent inputEvent)
	{
		if (_content == null)
		{
			return;
		}
		if (!(inputEvent is InputEventMouseMotion inputEventMouseMotion))
		{
			if (!(inputEvent is InputEventMouseButton inputEventMouseButton))
			{
				return;
			}
			if (inputEventMouseButton.ButtonIndex == MouseButton.Left)
			{
				_isDragging = inputEventMouseButton.Pressed;
				if (inputEventMouseButton.Pressed)
				{
					_startDragPosY = _content.Position.Y - _paddingTop;
					_targetDragPosY = _startDragPosY;
				}
			}
			else if (!inputEventMouseButton.Pressed)
			{
				_isDragging = false;
			}
		}
		else if (_isDragging)
		{
			_targetDragPosY += inputEventMouseMotion.Relative.Y;
		}
	}

	private void ProcessScrollEvent(InputEvent inputEvent)
	{
		_targetDragPosY += ScrollHelper.GetDragForScrollEvent(inputEvent);
	}

	public override void _Process(double delta)
	{
		if (IsVisibleInTree() && (!_disableScrollingIfContentFits || Scrollbar.Visible))
		{
			UpdateScrollPosition(delta);
		}
	}

	public void InstantlyScrollToTop()
	{
		if (_content == null)
		{
			throw new InvalidOperationException("No content to scroll!");
		}
		_targetDragPosY = 0f;
		Control? content = _content;
		Vector2 position = _content.Position;
		position.Y = _paddingTop;
		content.Position = position;
		Scrollbar.SetValueWithoutAnimation(0.0);
	}

	private void ProcessGuiFocus(Control focusedControl)
	{
		if (_content != null && IsVisibleInTree() && NControllerManager.Instance.IsUsingController && !(focusedControl is NDropdownItem) && _content.IsAncestorOf(focusedControl))
		{
			float num = _content.GlobalPosition.Y - focusedControl.GlobalPosition.Y;
			float value = num + ScrollViewportSize * 0.5f;
			float max = Mathf.Max(ScrollLimitBottom, 0f);
			float min = Mathf.Min(ScrollLimitBottom, 0f);
			value = Mathf.Clamp(value, min, max);
			_targetDragPosY = value;
		}
	}

	private void UpdateScrollPosition(double delta)
	{
		if (_content == null)
		{
			return;
		}
		float num = _paddingTop + _targetDragPosY;
		if (!Mathf.IsEqualApprox(_content.Position.Y, num))
		{
			float y = Mathf.Lerp(_content.Position.Y, num, (float)delta * 15f);
			Control? content = _content;
			Vector2 position = _content.Position;
			position.Y = y;
			content.Position = position;
			if (Mathf.Abs(_content.Position.Y - num) < 0.5f)
			{
				Control? content2 = _content;
				position = _content.Position;
				position.Y = num;
				content2.Position = position;
			}
			if (!_scrollbarPressed && ScrollLimitBottom < 0f)
			{
				Scrollbar.SetValueWithoutAnimation(Mathf.Clamp((_content.Position.Y - _paddingTop) / ScrollLimitBottom, 0f, 1f) * 100f);
			}
		}
		if (_scrollbarPressed)
		{
			_targetDragPosY = Mathf.Lerp(0f, ScrollLimitBottom, (float)Scrollbar.Value * 0.01f);
		}
		if (!_isDragging)
		{
			if (_targetDragPosY < Mathf.Min(ScrollLimitBottom, 0f))
			{
				_targetDragPosY = Mathf.Lerp(_targetDragPosY, ScrollLimitBottom, (float)delta * 12f);
			}
			else if (_targetDragPosY > Mathf.Max(ScrollLimitBottom, 0f))
			{
				_targetDragPosY = Mathf.Lerp(_targetDragPosY, 0f, (float)delta * 12f);
			}
		}
	}
}
