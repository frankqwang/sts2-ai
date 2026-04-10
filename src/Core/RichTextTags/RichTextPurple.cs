using Godot;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.RichTextTags;

[GlobalClass]
[Tool]
public partial class RichTextPurple : AbstractMegaRichTextEffect
{
	public new string bbcode = "purple";

	protected override string Bbcode => bbcode;

	public override bool _ProcessCustomFX(CharFXTransform charFx)
	{
		charFx.Color = StsColors.purple;
		return true;
	}
}
