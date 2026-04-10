using Godot;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.RichTextTags;

[GlobalClass]
[Tool]
public partial class RichTextPink : AbstractMegaRichTextEffect
{
	public new string bbcode = "pink";

	protected override string Bbcode => bbcode;

	public override bool _ProcessCustomFX(CharFXTransform charFx)
	{
		charFx.Color = StsColors.pink;
		return true;
	}
}
