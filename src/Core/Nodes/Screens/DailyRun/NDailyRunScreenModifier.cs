using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;

namespace MegaCrit.Sts2.Core.Nodes.Screens.DailyRun;

public partial class NDailyRunScreenModifier : Control
{
	private static readonly LocString _modifierLoc = new LocString("main_menu_ui", "DAILY_RUN_MENU.MODIFIER");

	private TextureRect _icon;

	private MegaRichTextLabel _description;

	public override void _Ready()
	{
		_icon = GetNode<TextureRect>("Icon");
		_description = GetNode<MegaRichTextLabel>("Description");
	}

	public void Fill(ModifierModel modifier)
	{
		_modifierLoc.Add("title", modifier.Title.GetFormattedText());
		_modifierLoc.Add("description", modifier.Description.GetFormattedText());
		_icon.Texture = modifier.Icon;
		_description.Text = _modifierLoc.GetFormattedText();
	}
}
