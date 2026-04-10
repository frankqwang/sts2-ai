using Godot;

namespace MegaCrit.Sts2.Core.RichTextTags;

[GlobalClass]
[Tool]
public partial class RichTextAncientBanner : AbstractMegaRichTextEffect
{
	public new string bbcode = "ancient_banner";

	public float Rotation { get; set; }

	public float Spacing { get; set; }

	public float CenterCharacter { get; set; }

	protected override string Bbcode => bbcode;

	public override bool _ProcessCustomFX(CharFXTransform charFx)
	{
		if (ShouldTransformText())
		{
			float num = (float)charFx.RelativeIndex + 0.5f - CenterCharacter;
			Transform2D transform = charFx.Transform;
			Vector2 x = charFx.Transform.X;
			x.X = Rotation;
			transform.X = x;
			charFx.Transform = transform;
			transform = charFx.Transform;
			x = charFx.Transform.Origin;
			x.X = charFx.Transform.Origin.X + num * Spacing;
			transform.Origin = x;
			charFx.Transform = transform;
		}
		else
		{
			double num2 = charFx.ElapsedTime * 3.0 - (double)((float)charFx.RelativeIndex * 0.015f);
			Color color = charFx.Color;
			color.A = Mathf.Clamp((float)num2, 0f, 1f);
			charFx.Color = color;
		}
		return true;
	}
}
