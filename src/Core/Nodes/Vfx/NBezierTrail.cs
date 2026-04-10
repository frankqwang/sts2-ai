using System.Collections.Generic;
using Godot;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NBezierTrail : Line2D
{
	private Node2D _target;

	private float _pointDuration = 0.5f;

	private readonly List<float> _pointAge = new List<float>();

	private const float _minSpawnDist = 12f;

	private const float _maxSpawnDist = 48f;

	private Vector2? _lastPointPosition;

	public override void _Ready()
	{
		_target = GetParent<Node2D>();
		Connect(CanvasItem.SignalName.VisibilityChanged, Callable.From(OnToggleVisibility));
	}

	public override void _Process(double delta)
	{
		base.GlobalPosition = Vector2.Zero;
		base.GlobalRotation = 0f;
		CreatePoint(_target.GlobalPosition, delta);
		float num = (float)delta;
		for (int i = 0; i < base.Points.Length; i++)
		{
			if (_pointAge[i] > _pointDuration)
			{
				RemovePoint(0);
				_pointAge.RemoveAt(0);
			}
			else
			{
				_pointAge[i] += num;
			}
		}
	}

	private void OnToggleVisibility()
	{
		base.ProcessMode = (ProcessModeEnum)(base.Visible ? 0 : 4);
		ClearPoints();
	}

	private void CreatePoint(Vector2 pointPos, double delta)
	{
		if (_lastPointPosition.HasValue)
		{
			float num = pointPos.DistanceTo(_lastPointPosition.Value);
			if (num < 12f)
			{
				return;
			}
			int pointCount = GetPointCount();
			if (pointCount > 2 && num > 48f)
			{
				Vector2 pointPosition = GetPointPosition(pointCount - 2);
				Vector2 pointPosition2 = GetPointPosition(pointCount - 1);
				for (float num2 = 48f; num2 < num - 12f; num2 += 48f)
				{
					float num3 = 0.5f + num2 / num * 0.5f;
					Vector2 vector = pointPosition.Lerp(pointPosition2, num3);
					Vector2 to = pointPosition2.Lerp(pointPos, num3);
					Vector2 position = vector.Lerp(to, num3);
					_pointAge.Add((float)delta * num3);
					AddPoint(position);
				}
			}
		}
		_pointAge.Add(0f);
		AddPoint(pointPos);
		_lastPointPosition = pointPos;
	}
}
