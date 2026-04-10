using Godot;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.RichTextTags;

[GlobalClass]
[Tool]
public partial class RichTextGold : AbstractMegaRichTextEffect
{
	public new string bbcode = "gold";

	protected override string Bbcode => bbcode;

	public override bool _ProcessCustomFX(CharFXTransform charFx)
	{
		if (charFx.Color == StsColors.green)
		{
			return true;
		}
		charFx.Color = StsColors.gold;
		return true;
	}
}
