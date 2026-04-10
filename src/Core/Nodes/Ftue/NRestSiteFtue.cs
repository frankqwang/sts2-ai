using System;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Ftue;

public partial class NRestSiteFtue : NFtue
{
	public const string id = "rest_site_ftue";

	private static readonly string _scenePath = SceneHelper.GetScenePath("ftue/rest_site_ftue");

	private NButton _confirmButton;

	private MegaLabel _header;

	private MegaRichTextLabel _description;

	private Control _arrow;

	private int _defaultZIndex;

	private Control _restSiteOptionContainer;

	public override void _Ready()
	{
		_header = GetNode<MegaLabel>("FtuePopup/Header");
		_header.SetTextAutoSize(new LocString("ftues", "REST_SITE_FTUE_TITLE").GetFormattedText());
		_description = GetNode<MegaRichTextLabel>("FtuePopup/DescriptionContainer/Description");
		_description.Text = new LocString("ftues", "REST_SITE_FTUE_DESCRIPTION").GetFormattedText();
		_confirmButton = GetNode<NButton>("FtuePopup/FtueConfirmButton");
		_confirmButton.Connect(NClickableControl.SignalName.Released, Callable.From((Action<NButton>)CloseFtue));
		_defaultZIndex = _restSiteOptionContainer.ZIndex;
		_restSiteOptionContainer.ZIndex++;
	}

	public static NRestSiteFtue? Create(Control restSiteOptionContainer)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NRestSiteFtue nRestSiteFtue = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NRestSiteFtue>(PackedScene.GenEditState.Disabled);
		nRestSiteFtue._restSiteOptionContainer = restSiteOptionContainer;
		return nRestSiteFtue;
	}

	private void CloseFtue(NButton _)
	{
		_restSiteOptionContainer.ZIndex = _defaultZIndex;
		CloseFtue();
	}
}
