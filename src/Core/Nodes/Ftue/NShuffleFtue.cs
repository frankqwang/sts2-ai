using System;
using System.Threading.Tasks;
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

public partial class NShuffleFtue : NFtue
{
	public const string id = "shuffle_ftue";

	private static readonly string _scenePath = SceneHelper.GetScenePath("ftue/shuffle_ftue");

	private NButton _confirmButton;

	private MegaLabel _header;

	private MegaRichTextLabel _description;

	private int _defaultZIndex;

	public override void _Ready()
	{
		_header = GetNode<MegaLabel>("FtuePopup/Header");
		_header.SetTextAutoSize(new LocString("ftues", "SHUFFLE_FTUE_TITLE").GetFormattedText());
		_description = GetNode<MegaRichTextLabel>("FtuePopup/DescriptionContainer/Description");
		_description.Text = new LocString("ftues", "SHUFFLE_FTUE_DESCRIPTION").GetFormattedText();
		_confirmButton = GetNode<NButton>("FtuePopup/FtueConfirmButton");
		_confirmButton.Connect(NClickableControl.SignalName.Released, Callable.From((Action<NButton>)CloseFtue));
		NCombatUi ui = NCombatRoom.Instance.Ui;
		_defaultZIndex = ui.DrawPile.ZIndex;
		ui.DrawPile.ZIndex++;
		ui.DiscardPile.ZIndex++;
	}

	public static NShuffleFtue? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NShuffleFtue>(PackedScene.GenEditState.Disabled);
	}

	public async Task WaitForPlayerToConfirm()
	{
		await ToSignal(_confirmButton, NClickableControl.SignalName.Released);
	}

	private void CloseFtue(NButton _)
	{
		NCombatUi ui = NCombatRoom.Instance.Ui;
		ui.DrawPile.ZIndex = _defaultZIndex;
		ui.DiscardPile.ZIndex = _defaultZIndex;
		CloseFtue();
	}
}
