using System;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Ftue;

public partial class NPowerCardFtue : NFtue
{
	public const string id = "power_card_ftue";

	private static readonly string _scenePath = SceneHelper.GetScenePath("ftue/power_card_ftue");

	private NButton _confirmButton;

	private MegaLabel _header;

	private MegaRichTextLabel _description;

	private Control _card;

	private Control _ftueHolder;

	private int _defaultZIndex;

	public override void _Ready()
	{
		_header = GetNode<MegaLabel>("FtuePopup/Header");
		_header.SetTextAutoSize(new LocString("ftues", "POWER_FTUE_TITLE").GetFormattedText());
		_description = GetNode<MegaRichTextLabel>("FtuePopup/DescriptionContainer/Description");
		_description.Text = new LocString("ftues", "POWER_FTUE_DESCRIPTION").GetFormattedText();
		_confirmButton = GetNode<NButton>("FtuePopup/FtueConfirmButton");
		_confirmButton.Connect(NClickableControl.SignalName.Released, Callable.From((Action<NButton>)CloseFtue));
		_ftueHolder = GetNode<Control>("FtuePopup");
		_defaultZIndex = _card.ZIndex;
		_card.ZIndex++;
	}

	public static NPowerCardFtue? Create(Control card)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NPowerCardFtue nPowerCardFtue = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NPowerCardFtue>(PackedScene.GenEditState.Disabled);
		nPowerCardFtue._card = card;
		return nPowerCardFtue;
	}

	private void CloseFtue(NButton _)
	{
		_card.ZIndex = _defaultZIndex;
		CloseFtue();
	}
}
