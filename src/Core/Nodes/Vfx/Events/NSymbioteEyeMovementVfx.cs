using Godot;

namespace MegaCrit.Sts2.Core.Nodes.Vfx.Events;

public partial class NSymbioteEyeMovementVfx : PathFollow2D
{
	private const float _speed = 0.1f;

	public override void _Process(double delta)
	{
		base.ProgressRatio += (float)delta * 0.1f;
	}
}
