using System;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Ftue;

public partial class NCannotPlayCardFtue : NFtue
{
	public const string id = "cannot_play_card_ftue";

	private static readonly string _scenePath = SceneHelper.GetScenePath("ftue/cannot_play_card_ftue");

	private NButton _confirmButton;

	private MegaLabel _header;

	private MegaRichTextLabel _description;

	private Control _sneakyHitbox;

	public override void _Ready()
	{
		_header = GetNode<MegaLabel>("FtuePopup/Header");
		_header.SetTextAutoSize(new LocString("ftues", "CANNOT_PLAY_CARD_FTUE_TITLE").GetFormattedText());
		_description = GetNode<MegaRichTextLabel>("FtuePopup/DescriptionContainer/Description");
		_description.Text = new LocString("ftues", "CANNOT_PLAY_CARD_FTUE_DESCRIPTION").GetFormattedText();
		_sneakyHitbox = GetNode<Control>("SneakyHitbox");
		_sneakyHitbox.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(CloseFtueAndEndTurn));
		_confirmButton = GetNode<NButton>("FtuePopup/FtueConfirmButton");
		_confirmButton.Connect(NClickableControl.SignalName.Released, Callable.From((Action<NButton>)CloseFtue));
		NEndTurnButton endTurnButton = NCombatRoom.Instance.Ui.EndTurnButton;
		endTurnButton.ZIndex = 1;
		_sneakyHitbox.GlobalPosition = endTurnButton.GlobalPosition;
	}

	public static NCannotPlayCardFtue? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NCannotPlayCardFtue>(PackedScene.GenEditState.Disabled);
	}

	private void CloseFtueAndEndTurn(NButton _)
	{
		NCombatRoom.Instance.Ui.EndTurnButton.SecretEndTurnLogicViaFtue();
		CloseFtue(_);
	}

	private void CloseFtue(NButton _)
	{
		NCombatRoom.Instance.Ui.EndTurnButton.ZIndex = 0;
		CloseFtue();
	}
}
