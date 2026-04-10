using Godot;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Timeline;

public partial class NTimelineLineMask : Control
{
	public override void _Process(double delta)
	{
		base._Process(delta);
		base.Position = Vector2.Zero;
	}
}
