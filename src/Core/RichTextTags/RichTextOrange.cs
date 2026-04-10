using Godot;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.RichTextTags;

[GlobalClass]
[Tool]
public partial class RichTextOrange : AbstractMegaRichTextEffect
{
	public new string bbcode = "orange";

	protected override string Bbcode => bbcode;

	public override bool _ProcessCustomFX(CharFXTransform charFx)
	{
		charFx.Color = StsColors.orange;
		return true;
	}
}
