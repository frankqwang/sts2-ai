using Godot;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NFollowCursor : Node2D
{
	public override void _Process(double delta)
	{
		Vector2 mousePosition = GetViewport().GetMousePosition();
		base.GlobalPosition = mousePosition;
	}
}
