using Godot;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;

public partial class NVfxParticleSystem : Node2D
{
	[Export(PropertyHint.None, "")]
	private float _lifetime = 1f;

	public override void _Ready()
	{
		foreach (Node child in GetChildren())
		{
			if (!(child is CpuParticles2D cpuParticles2D))
			{
				if (child is GpuParticles2D gpuParticles2D)
				{
					gpuParticles2D.Emitting = true;
				}
			}
			else
			{
				cpuParticles2D.Emitting = true;
			}
		}
		SceneTreeTimer sceneTreeTimer = GetTree().CreateTimer(_lifetime);
		sceneTreeTimer.Connect(SceneTreeTimer.SignalName.Timeout, Callable.From(AfterExpired));
	}

	private void AfterExpired()
	{
		if (GodotObject.IsInstanceValid(this))
		{
			this.QueueFreeSafely();
		}
	}
}
