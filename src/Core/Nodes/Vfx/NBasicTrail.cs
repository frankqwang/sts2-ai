using Godot;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NBasicTrail : Line2D
{
	private Node2D _target;

	[Export(PropertyHint.None, "")]
	private int _maxSegments = 10;

	public override void _Ready()
	{
		_target = GetParent<Node2D>();
	}

	public override void _Process(double delta)
	{
		base.GlobalPosition = Vector2.Zero;
		base.GlobalRotation = 0f;
		base.GlobalScale = Vector2.One;
		AddPoint(_target.GlobalPosition);
		if (base.Points.Length > _maxSegments)
		{
			RemovePoint(0);
		}
	}
}
