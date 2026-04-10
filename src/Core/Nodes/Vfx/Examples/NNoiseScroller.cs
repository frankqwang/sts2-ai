using Godot;

namespace MegaCrit.Sts2.Core.Nodes.Vfx.Examples;

public partial class NNoiseScroller : TextureRect
{
	[Export(PropertyHint.None, "")]
	private Vector3 _offsetDelta;

	private FastNoiseLite _noise;

	public override void _Ready()
	{
		_noise = (FastNoiseLite)((NoiseTexture2D)base.Texture).Noise;
	}

	public override void _Process(double delta)
	{
		_noise.Offset += _offsetDelta * (float)delta;
	}
}
