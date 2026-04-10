using Godot;
using Godot.Collections;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.RichTextTags;

[GlobalClass]
[Tool]
public partial class RichTextFlyIn : AbstractMegaRichTextEffect
{
	private static readonly Variant _xOffsetKey = Variant.From<string>("offset_x");

	private static readonly Variant _yOffsetKey = Variant.From<string>("offset_y");

	public new string bbcode = "fly_in";

	protected override string Bbcode => bbcode;

	public override bool _ProcessCustomFX(CharFXTransform charFx)
	{
		if (Engine.IsEditorHint())
		{
			return false;
		}
		Dictionary env = charFx.Env;
		Vector2 zero = Vector2.Zero;
		if (env.TryGetValue(_xOffsetKey, out var value))
		{
			zero.X = (float)value.AsDouble();
		}
		if (env.TryGetValue(_yOffsetKey, out var value2))
		{
			zero.Y = (float)value2.AsDouble();
		}
		double num = charFx.ElapsedTime * 3.0 - (double)((float)charFx.RelativeIndex * 0.015f);
		Color color = charFx.Color;
		color.A = Mathf.Clamp((float)num, 0f, 1f);
		charFx.Color = color;
		if (ShouldTransformText())
		{
			Vector2 vector = new Vector2(charFx.Transform.X.X, charFx.Transform.Y.Y);
			Vector2 vector2 = zero.Lerp(vector, Ease.QuadOut(color.A));
			Vector2 offset = vector2 - vector;
			charFx.Transform = charFx.Transform.TranslatedLocal(offset);
			charFx.Transform = charFx.Transform.RotatedLocal(Ease.QuadOut(1f - color.A) * Mathf.DegToRad(20f) * ((offset.X < 0f) ? 1f : (-1f)));
		}
		return true;
	}
}
