using Godot;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.RichTextTags;

public abstract partial class AbstractMegaRichTextEffect : RichTextEffect
{
	public string bbcode => Bbcode;

	protected abstract string Bbcode { get; }

	protected bool ShouldTransformText()
	{
		if (Engine.IsEditorHint())
		{
			return true;
		}
		return SaveManager.Instance.PrefsSave.TextEffectsEnabled;
	}
}
