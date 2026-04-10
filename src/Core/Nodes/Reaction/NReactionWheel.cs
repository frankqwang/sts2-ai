using System;
using Godot;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Reaction;

public partial class NReactionWheel : Control
{
	private static readonly StringName _reactWheel = new StringName("react_wheel");

	private const float _centerRadius = 70f;

	private NReactionWheelWedge _rightWedge;

	private NReactionWheelWedge _downRightWedge;

	private NReactionWheelWedge _downWedge;

	private NReactionWheelWedge _downLeftWedge;

	private NReactionWheelWedge _leftWedge;

	private NReactionWheelWedge _upLeftWedge;

	private NReactionWheelWedge _upWedge;

	private NReactionWheelWedge _upRightWedge;

	private TextureRect _marker;

	private bool _ignoreNextMouseInput;

	private Vector2 _centerPosition;

	private NReactionWheelWedge? _selectedWedge;

	private Player? _localPlayer;

	public override void _Ready()
	{
		_rightWedge = GetNode<NReactionWheelWedge>("RightWedge");
		_downRightWedge = GetNode<NReactionWheelWedge>("DownRightWedge");
		_downWedge = GetNode<NReactionWheelWedge>("DownWedge");
		_downLeftWedge = GetNode<NReactionWheelWedge>("DownLeftWedge");
		_leftWedge = GetNode<NReactionWheelWedge>("LeftWedge");
		_upLeftWedge = GetNode<NReactionWheelWedge>("UpLeftWedge");
		_upWedge = GetNode<NReactionWheelWedge>("UpWedge");
		_upRightWedge = GetNode<NReactionWheelWedge>("UpRightWedge");
		_marker = GetNode<TextureRect>("Marker");
		base.Visible = false;
	}

	public override void _EnterTree()
	{
		RunManager.Instance.RunStarted += OnRunStarted;
	}

	public override void _ExitTree()
	{
		RunManager.Instance.RunStarted -= OnRunStarted;
	}

	public override void _Notification(int what)
	{
		if (base.Visible && (long)what == 2017)
		{
			base.Visible = false;
		}
	}

	public override void _Input(InputEvent inputEvent)
	{
		Control control = GetViewport().GuiGetFocusOwner();
		bool flag = ((control is TextEdit || control is LineEdit) ? true : false);
		bool flag2 = flag;
		if (!NGame.Instance.ReactionContainer.InMultiplayer)
		{
			if (base.Visible)
			{
				HideWheel();
			}
		}
		else if (inputEvent is InputEventMouseMotion inputEventMouseMotion)
		{
			if (_ignoreNextMouseInput)
			{
				_ignoreNextMouseInput = false;
			}
			else if (base.Visible)
			{
				MoveMarker(inputEventMouseMotion.Relative);
				_ignoreNextMouseInput = true;
				WarpMouseBackToOriginalPosition();
			}
		}
		else if (inputEvent.IsActionPressed(_reactWheel) && !flag2)
		{
			base.Visible = true;
			if (_localPlayer != null)
			{
				_marker.Texture = _localPlayer.Character.MapMarker;
			}
			_centerPosition = GetViewport().GetMousePosition();
			_marker.Position = (base.Size - _marker.Size) * 0.5f;
			base.GlobalPosition = _centerPosition - base.Size * base.Scale * 0.5f;
			Input.MouseMode = Input.MouseModeEnum.Hidden;
		}
		else if (inputEvent.IsActionReleased(_reactWheel) && base.Visible)
		{
			HideWheel();
			React();
		}
	}

	private void OnRunStarted(RunState runState)
	{
		_localPlayer = LocalContext.GetMe(runState);
	}

	private void HideWheel()
	{
		Input.MouseMode = Input.MouseModeEnum.Visible;
		WarpMouseBackToOriginalPosition();
		base.Visible = false;
	}

	private void WarpMouseBackToOriginalPosition()
	{
		Transform2D viewportTransform = GetViewportTransform();
		Input.WarpMouse(viewportTransform * _centerPosition);
	}

	private void React()
	{
		if (_selectedWedge != null)
		{
			NGame.Instance.ReactionContainer.DoLocalReaction(_selectedWedge.Reaction, _centerPosition);
		}
	}

	private void MoveMarker(Vector2 relative)
	{
		Vector2 vector = (base.Size - _marker.Size) * 0.5f;
		Vector2 vector2 = _marker.Position - vector;
		vector2 = (vector2 + relative).LimitLength(70f);
		_marker.Position = vector + vector2;
		float num = Mathf.Atan2(vector2.Y, vector2.X);
		_marker.Rotation = num - (float)Math.PI / 2f;
		NReactionWheelWedge selectedWedge = GetSelectedWedge(num);
		if (_selectedWedge != selectedWedge)
		{
			_selectedWedge?.OnDeselected();
			_selectedWedge = selectedWedge;
			_selectedWedge?.OnSelected();
		}
	}

	private NReactionWheelWedge GetSelectedWedge(float angle)
	{
		float num = Mathf.Wrap(angle + (float)Math.PI / 8f, 0f, (float)Math.PI * 2f);
		float num2 = (float)Math.PI / 4f;
		return (int)(num / num2) switch
		{
			0 => _rightWedge, 
			1 => _downRightWedge, 
			2 => _downWedge, 
			3 => _downLeftWedge, 
			4 => _leftWedge, 
			5 => _upLeftWedge, 
			6 => _upWedge, 
			7 => _upRightWedge, 
			_ => throw new InvalidOperationException(), 
		};
	}
}
