using Godot;

namespace MegaCrit.Sts2.Core.Nodes.RestSite;

public partial class NParticleSystemUpscaler : CpuParticles2D
{
	[Export(PropertyHint.None, "")]
	private Vector2 _originalResolution;

	public override void _Ready()
	{
		Texture2D texture = base.Texture;
		Vector2 size = texture.GetSize();
		float num = _originalResolution.X / size.X;
		base.ScaleAmountMin *= num;
		base.ScaleAmountMax = base.ScaleAmountMin;
	}
}
