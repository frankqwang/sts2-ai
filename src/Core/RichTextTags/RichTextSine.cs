using System;
using Godot;
using Godot.Collections;

namespace MegaCrit.Sts2.Core.RichTextTags;

[GlobalClass]
[Tool]
public partial class RichTextSine : AbstractMegaRichTextEffect
{
	private const float _amplitude = 0.8f;

	private const float _frequency = 0.5f;

	private const float _speed = 1.5f;

	public new string bbcode = "sine";

	protected override string Bbcode => bbcode;

	public override bool _ProcessCustomFX(CharFXTransform charFx)
	{
		if (!ShouldTransformText())
		{
			return false;
		}
		Dictionary env = charFx.Env;
		float num = (float)(charFx.ElapsedTime * 1.5 + (double)((float)charFx.RelativeIndex * 0.1f));
		float y = 0.8f * Mathf.Sin(num * (float)Math.PI * 2f * 0.5f);
		charFx.Offset += new Vector2(0f, y);
		if (env.TryGetValue(RichTextUtil.colorKey, out var value))
		{
			charFx.Color = (Color)value;
		}
		charFx.Visible = !env.ContainsKey(RichTextUtil.visibleKey) || (bool)env[RichTextUtil.visibleKey];
		return true;
	}
}
