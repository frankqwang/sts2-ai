using Godot;
using Godot.Collections;

namespace MegaCrit.Sts2.Core.RichTextTags;

[GlobalClass]
[Tool]
public partial class RichTextFadeIn : AbstractMegaRichTextEffect
{
	private static readonly Variant _speedKey = Variant.From<string>("speed");

	private static readonly Variant _tickKey = Variant.From<string>("tick");

	public new string bbcode = "fade_in";

	protected override string Bbcode => bbcode;

	public override bool _ProcessCustomFX(CharFXTransform charFx)
	{
		Dictionary env = charFx.Env;
		charFx.Offset = Vector2.Zero;
		Variant value;
		double num = ((!env.TryGetValue(_speedKey, out value)) ? 4.0 : value.AsDouble());
		Variant value2;
		double num2 = ((!env.TryGetValue(_tickKey, out value2)) ? 0.009999999776482582 : value2.AsDouble());
		double num3 = charFx.ElapsedTime * num - (double)charFx.RelativeIndex * num2;
		Color color = charFx.Color;
		color.A = Mathf.Clamp((float)num3, 0f, 1f);
		charFx.Color = color;
		if (env.TryGetValue(RichTextUtil.visibleKey, out var value3))
		{
			charFx.Visible = value3.AsBool();
		}
		else
		{
			charFx.Visible = true;
		}
		return true;
	}
}
