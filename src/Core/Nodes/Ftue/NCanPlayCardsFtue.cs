using System;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Ftue;

public partial class NCanPlayCardsFtue : NFtue
{
	public const string id = "can_play_cards_ftue";

	private static readonly string _scenePath = SceneHelper.GetScenePath("ftue/can_play_cards_ftue");

	private NButton _confirmButton;

	private MegaLabel _header;

	private MegaRichTextLabel _description;

	public override void _Ready()
	{
		_header = GetNode<MegaLabel>("FtuePopup/Header");
		_header.SetTextAutoSize(new LocString("ftues", "CAN_PLAY_CARDS_FTUE_TITLE").GetFormattedText());
		_description = GetNode<MegaRichTextLabel>("FtuePopup/DescriptionContainer/Description");
		_description.Text = new LocString("ftues", "CAN_PLAY_CARDS_FTUE_DESCRIPTION").GetFormattedText();
		_confirmButton = GetNode<NButton>("FtuePopup/FtueConfirmButton");
		_confirmButton.Connect(NClickableControl.SignalName.Released, Callable.From((Action<NButton>)CloseFtue));
		NCombatRoom.Instance.Ui.EnergyCounterContainer.ZIndex = 1;
	}

	public static NCanPlayCardsFtue? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NCanPlayCardsFtue>(PackedScene.GenEditState.Disabled);
	}

	private void CloseFtue(NButton _)
	{
		NCombatRoom.Instance.Ui.EnergyCounterContainer.ZIndex = 0;
		CloseFtue();
	}
}
