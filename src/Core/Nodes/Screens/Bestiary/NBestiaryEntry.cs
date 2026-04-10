using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Bestiary;

public partial class NBestiaryEntry : NButton
{
	private MegaRichTextLabel _nameLabel;

	private Control _highlight;

	private static string ScenePath => SceneHelper.GetScenePath("screens/bestiary/bestiary_entry");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	public MonsterModel Monster { get; private set; }

	public bool IsLocked { get; private set; }

	public static NBestiaryEntry Create(MonsterModel monster, bool isLocked)
	{
		NBestiaryEntry nBestiaryEntry = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NBestiaryEntry>(PackedScene.GenEditState.Disabled);
		nBestiaryEntry.Monster = monster;
		nBestiaryEntry.IsLocked = isLocked;
		return nBestiaryEntry;
	}

	public override void _Ready()
	{
		ConnectSignals();
		_nameLabel = GetNode<MegaRichTextLabel>("Label");
		_highlight = GetNode<Control>("Highlight");
		if (IsLocked)
		{
			_nameLabel.Text = "[Locked]";
			_nameLabel.Modulate = StsColors.gray;
		}
		else
		{
			_nameLabel.Text = Monster.Title.GetFormattedText();
			_nameLabel.Modulate = StsColors.cream;
		}
	}

	public void Select()
	{
		_nameLabel.Modulate = StsColors.gold;
	}

	public void Deselect()
	{
		_nameLabel.Modulate = (IsLocked ? StsColors.gray : StsColors.cream);
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
