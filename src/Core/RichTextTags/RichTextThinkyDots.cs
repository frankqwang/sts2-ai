using System;
using Godot;
using Godot.Collections;

namespace MegaCrit.Sts2.Core.RichTextTags;

[GlobalClass]
[Tool]
public partial class RichTextThinkyDots : AbstractMegaRichTextEffect
{
	private const float _amplitude = 1.5f;

	private const float _frequency = 0.4f;

	private const float _speed = 1f;

	private const float _spacing = 4f;

	public new string bbcode = "thinky_dots";

	protected override string Bbcode => bbcode;

	public override bool _ProcessCustomFX(CharFXTransform charFx)
	{
		if (!ShouldTransformText())
		{
			return false;
		}
		Dictionary env = charFx.Env;
		charFx.Offset = Vector2.Zero;
		float val = (float)(charFx.ElapsedTime * 1.0 - (double)((float)charFx.RelativeIndex * 0.1f));
		val = Math.Max(val, 0f);
		float num = val % 4.4f;
		float a = ((!(num < 0.4f)) ? 0f : (1.5f * Mathf.Sin(num / 0.4f * (float)Math.PI)));
		charFx.Offset += new Vector2(0f, 0f - Mathf.Max(a, 0f));
		if (env.TryGetValue(RichTextUtil.colorKey, out var value))
		{
			charFx.Color = (Color)value;
		}
		charFx.Visible = !env.ContainsKey(RichTextUtil.visibleKey) || (bool)env[RichTextUtil.visibleKey];
		return true;
	}
}
