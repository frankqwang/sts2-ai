using System.Collections.Generic;
using System.Linq;
using Godot;

namespace MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;

public partial class NTrail2D : Line2D
{
	private int _maxSegments = 20;

	private Node2D _parent;

	private readonly List<Vector2> _pointQueue = new List<Vector2>();

	private bool _isActive;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		Connect(CanvasItem.SignalName.VisibilityChanged, Callable.From(OnToggleVisibility));
		OnToggleVisibility();
	}

	private void OnToggleVisibility()
	{
		_isActive = base.Visible;
		if (!base.Visible)
		{
			_pointQueue.Clear();
			base.Points = _pointQueue.ToArray();
		}
	}

	public override void _Process(double delta)
	{
		if (_isActive)
		{
			_pointQueue.Insert(0, _parent.GlobalPosition);
			if (_pointQueue.Count >= _maxSegments)
			{
				_pointQueue.RemoveAt(_pointQueue.Count - 1);
			}
			base.Points = _pointQueue.Select((Vector2 point) => GetParent<Node2D>().ToLocal(point)).ToArray();
		}
	}
}
