using System.Linq;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens.CustomRun;

public partial class NRunModifierTickbox : NTickbox
{
	public const string scenePath = "res://scenes/screens/custom_run/modifier_tickbox.tscn";

	private static readonly LocString _descriptionLoc = new LocString("main_menu_ui", "CUSTOM_RUN_SCREEN.MODIFIER_LABEL");

	private MegaRichTextLabel _label;

	private Control _highlight;

	public ModifierModel? Modifier { get; private set; }

	public static NRunModifierTickbox? Create(ModifierModel model)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NRunModifierTickbox nRunModifierTickbox = PreloadManager.Cache.GetScene("res://scenes/screens/custom_run/modifier_tickbox.tscn").Instantiate<NRunModifierTickbox>(PackedScene.GenEditState.Disabled);
		nRunModifierTickbox.Modifier = model;
		return nRunModifierTickbox;
	}

	protected override void OnEnable()
	{
		base.OnEnable();
		base.Modulate = Colors.White;
	}

	protected override void OnDisable()
	{
		base.OnDisable();
		base.Modulate = Colors.Gray;
	}

	public override void _Ready()
	{
		ConnectSignals();
		_label = GetNode<MegaRichTextLabel>("%Description");
		_highlight = GetNode<Control>("Highlight");
		if (Modifier != null)
		{
			string variable = ((ModelDb.GoodModifiers.FirstOrDefault((ModifierModel m) => m.GetType() == Modifier.GetType()) != null) ? "green" : ((ModelDb.BadModifiers.FirstOrDefault((ModifierModel m) => m.GetType() == Modifier.GetType()) != null) ? "red" : "blue"));
			_descriptionLoc.Add("color", variable);
			_descriptionLoc.Add("modifier_title", Modifier.Title.GetFormattedText());
			_descriptionLoc.Add("modifier_description", Modifier.Description.GetFormattedText());
			_label.Text = _descriptionLoc.GetFormattedText();
		}
		base.IsTicked = false;
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		if (NControllerManager.Instance.IsUsingController)
		{
			_highlight.Visible = true;
		}
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		_highlight.Visible = false;
	}
}
