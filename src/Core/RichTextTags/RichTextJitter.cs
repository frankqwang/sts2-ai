using Godot;
using Godot.Collections;

namespace MegaCrit.Sts2.Core.RichTextTags;

[GlobalClass]
[Tool]
public partial class RichTextJitter : AbstractMegaRichTextEffect
{
	private const float _amplitude = 3f;

	private const float _speed = 600f;

	public new string bbcode = "jitter";

	private FastNoiseLite _fastNoise;

	protected override string Bbcode => bbcode;

	public RichTextJitter()
	{
		_fastNoise = new FastNoiseLite();
		_fastNoise.NoiseType = FastNoiseLite.NoiseTypeEnum.Perlin;
		_fastNoise.FractalOctaves = 8;
		_fastNoise.FractalGain = 0.8f;
	}

	public override bool _ProcessCustomFX(CharFXTransform charFx)
	{
		Dictionary env = charFx.Env;
		_fastNoise.Seed = (charFx.RelativeIndex + 1) * 131;
		float noise1D = _fastNoise.GetNoise1D((float)charFx.ElapsedTime * 600f);
		_fastNoise.Seed = (charFx.RelativeIndex + 1) * 737;
		float noise1D2 = _fastNoise.GetNoise1D((float)charFx.ElapsedTime * 600f);
		charFx.Offset += new Vector2(noise1D, noise1D2) * 3f;
		if (env.TryGetValue(RichTextUtil.colorKey, out var value))
		{
			charFx.Color = (Color)value;
		}
		charFx.Visible = !env.ContainsKey(RichTextUtil.visibleKey) || (bool)env[RichTextUtil.visibleKey];
		return true;
	}
}
