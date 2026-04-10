using System;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Ftue;

public partial class NRelicRewardFtue : NFtue
{
	public const string id = "obtain_relic_ftue";

	private static readonly string _scenePath = SceneHelper.GetScenePath("ftue/obtain_relic_ftue");

	private NButton _confirmButton;

	private MegaLabel _header;

	private MegaRichTextLabel _description;

	private Control _arrow;

	private int _defaultZIndex;

	private Control _relicReward;

	public override void _Ready()
	{
		_header = GetNode<MegaLabel>("FtuePopup/Header");
		_header.SetTextAutoSize(new LocString("ftues", "RELIC_FTUE_TITLE").GetFormattedText());
		_description = GetNode<MegaRichTextLabel>("FtuePopup/DescriptionContainer/Description");
		_description.Text = new LocString("ftues", "RELIC_FTUE_DESCRIPTION").GetFormattedText();
		_confirmButton = GetNode<NButton>("FtuePopup/FtueConfirmButton");
		_confirmButton.Connect(NClickableControl.SignalName.Released, Callable.From((Action<NButton>)CloseFtue));
		_arrow = GetNode<Control>("%Arrow");
		_arrow.GlobalPosition = _relicReward.GlobalPosition + new Vector2(_relicReward.Size.X * 0.5f, _relicReward.Size.Y);
		Log.Info($"{_relicReward.GlobalPosition} {_relicReward.Size} {_arrow.GlobalPosition}");
		_defaultZIndex = _relicReward.ZIndex;
		_relicReward.ZIndex++;
	}

	public static NRelicRewardFtue? Create(Control relicReward)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NRelicRewardFtue nRelicRewardFtue = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NRelicRewardFtue>(PackedScene.GenEditState.Disabled);
		nRelicRewardFtue._relicReward = relicReward;
		return nRelicRewardFtue;
	}

	private void CloseFtue(NButton _)
	{
		_relicReward.ZIndex = _defaultZIndex;
		CloseFtue();
	}
}
