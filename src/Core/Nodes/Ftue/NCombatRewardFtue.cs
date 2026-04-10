using System;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Ftue;

public partial class NCombatRewardFtue : NFtue
{
	public const string id = "combat_reward_ftue";

	private static readonly string _scenePath = SceneHelper.GetScenePath("ftue/combat_reward_ftue");

	private NButton _confirmButton;

	private MegaLabel _header;

	private MegaRichTextLabel _description;

	private Control _rewardsContainer;

	private int _defaultZIndex;

	public override void _Ready()
	{
		_header = GetNode<MegaLabel>("FtuePopup/Header");
		_header.SetTextAutoSize(new LocString("ftues", "REWARDS_FTUE_TITLE").GetFormattedText());
		_description = GetNode<MegaRichTextLabel>("FtuePopup/DescriptionContainer/Description");
		_description.Text = new LocString("ftues", "REWARDS_FTUE_DESCRIPTION").GetFormattedText();
		_confirmButton = GetNode<NButton>("FtuePopup/FtueConfirmButton");
		_confirmButton.Connect(NClickableControl.SignalName.Released, Callable.From((Action<NButton>)CloseFtue));
		_defaultZIndex = _rewardsContainer.ZIndex;
		_rewardsContainer.ZIndex++;
	}

	public static NCombatRewardFtue? Create(Control rewardsContainer)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCombatRewardFtue nCombatRewardFtue = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NCombatRewardFtue>(PackedScene.GenEditState.Disabled);
		nCombatRewardFtue._rewardsContainer = rewardsContainer;
		return nCombatRewardFtue;
	}

	private void CloseFtue(NButton _)
	{
		_rewardsContainer.ZIndex = _defaultZIndex;
		CloseFtue();
	}

	public async Task WaitForPlayerToConfirm()
	{
		await ToSignal(_confirmButton, NClickableControl.SignalName.Released);
	}
}
