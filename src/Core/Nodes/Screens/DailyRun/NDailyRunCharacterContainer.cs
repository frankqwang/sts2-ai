using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Platform;

namespace MegaCrit.Sts2.Core.Nodes.Screens.DailyRun;

public partial class NDailyRunCharacterContainer : Control
{
	private static readonly LocString _ascensionLoc = new LocString("main_menu_ui", "DAILY_RUN_MENU.ASCENSION");

	private Control _characterIconContainer;

	private MegaLabel _playerNameLabel;

	private MegaLabel _characterNameLabel;

	private MegaLabel _ascensionLabel;

	private MegaLabel _ascensionNumberLabel;

	private Control _readyIndicator;

	public override void _Ready()
	{
		_characterIconContainer = GetNode<Control>("%CharacterIconContainer");
		_playerNameLabel = GetNode<MegaLabel>("%PlayerNameLabel");
		_characterNameLabel = GetNode<MegaLabel>("%CharacterNameLabel");
		_ascensionLabel = GetNode<MegaLabel>("%AscensionLabel");
		_ascensionNumberLabel = GetNode<MegaLabel>("%AscensionNumberLabel");
		_readyIndicator = GetNode<Control>("%ReadyIndicator");
	}

	public void Fill(CharacterModel character, ulong playerId, int ascension, INetGameService netService)
	{
		_ascensionLoc.Add("ascension", ascension);
		bool flag = netService.Type.IsMultiplayer();
		Control icon = character.Icon;
		foreach (Node child in _characterIconContainer.GetChildren())
		{
			_characterIconContainer.RemoveChildSafely(child);
		}
		_playerNameLabel.Visible = flag;
		_characterNameLabel.Modulate = (flag ? StsColors.cream : StsColors.gold);
		_characterIconContainer.AddChildSafely(icon);
		_characterNameLabel.SetTextAutoSize(character.Title.GetFormattedText());
		_playerNameLabel.SetTextAutoSize(PlatformUtil.GetPlayerName(netService.Platform, playerId));
		_ascensionLabel.SetTextAutoSize(_ascensionLoc.GetFormattedText());
		_ascensionNumberLabel.SetTextAutoSize(ascension.ToString());
	}

	public void SetIsReady(bool isReady)
	{
		_readyIndicator.Visible = isReady;
	}
}
