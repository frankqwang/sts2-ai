using System;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Ftue;

public partial class NObtainPotionFtue : NFtue
{
	public const string id = "obtain_potion_ftue";

	private static readonly string _scenePath = SceneHelper.GetScenePath("ftue/obtain_potion_ftue");

	private NButton _confirmButton;

	private MegaLabel _header;

	private MegaRichTextLabel _description;

	private int _defaultZIndex;

	public override void _Ready()
	{
		_header = GetNode<MegaLabel>("FtuePopup/Header");
		_header.SetTextAutoSize(new LocString("ftues", "POTION_FTUE_TITLE").GetFormattedText());
		_description = GetNode<MegaRichTextLabel>("FtuePopup/DescriptionContainer/Description");
		_description.Text = new LocString("ftues", "POTION_FTUE_DESCRIPTION").GetFormattedText();
		_confirmButton = GetNode<NButton>("FtuePopup/FtueConfirmButton");
		_confirmButton.Connect(NClickableControl.SignalName.Released, Callable.From((Action<NButton>)CloseFtue));
		_defaultZIndex = NRun.Instance.GlobalUi.TopBar.PotionContainer.ZIndex;
		NRun.Instance.GlobalUi.TopBar.PotionContainer.ZIndex++;
	}

	public static NObtainPotionFtue? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NObtainPotionFtue>(PackedScene.GenEditState.Disabled);
	}

	private void CloseFtue(NButton _)
	{
		NRun.Instance.GlobalUi.TopBar.PotionContainer.ZIndex = _defaultZIndex;
		CloseFtue();
	}
}
