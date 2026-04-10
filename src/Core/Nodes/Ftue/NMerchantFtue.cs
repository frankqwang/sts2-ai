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

public partial class NMerchantFtue : NFtue
{
	public const string id = "merchant_ftue";

	private static readonly string _scenePath = SceneHelper.GetScenePath("ftue/merchant_ftue");

	private NButton _confirmButton;

	private MegaLabel _header;

	private MegaRichTextLabel _description;

	private Control _sneakyHitbox;

	private NMerchantRoom _merchantRoom;

	private int _defaultZIndex;

	public override void _Ready()
	{
		_header = GetNode<MegaLabel>("FtuePopup/Header");
		_header.SetTextAutoSize(new LocString("ftues", "MERCHANT_FTUE_TITLE").GetFormattedText());
		_description = GetNode<MegaRichTextLabel>("FtuePopup/DescriptionContainer/Description");
		_description.Text = new LocString("ftues", "MERCHANT_FTUE_DESCRIPTION").GetFormattedText();
		_confirmButton = GetNode<NButton>("FtuePopup/FtueConfirmButton");
		_confirmButton.Connect(NClickableControl.SignalName.Released, Callable.From((Action<NButton>)CloseFtue));
		_sneakyHitbox = GetNode<Control>("SneakyHitbox");
		_sneakyHitbox.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(CloseFtueAndOpenRug));
		_defaultZIndex = _merchantRoom.MerchantButton.ZIndex;
		_merchantRoom.MerchantButton.ZIndex++;
	}

	public static NMerchantFtue? Create(NMerchantRoom merchantRoom)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NMerchantFtue nMerchantFtue = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NMerchantFtue>(PackedScene.GenEditState.Disabled);
		nMerchantFtue._merchantRoom = merchantRoom;
		return nMerchantFtue;
	}

	private void CloseFtueAndOpenRug(NButton _)
	{
		_merchantRoom.OpenInventory();
		CloseFtue(_);
	}

	private void CloseFtue(NButton _)
	{
		_merchantRoom.MerchantButton.ZIndex = _defaultZIndex;
		CloseFtue();
	}
}
