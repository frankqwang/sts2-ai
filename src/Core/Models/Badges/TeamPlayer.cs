using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Saves.Runs;

namespace MegaCrit.Sts2.Core.Models.Badges;

public class TeamPlayer : Badge
{
	public override string Id => "TEAM_PLAYER";

	public override BadgeRarity Rarity => BadgeRarity.Silver;

	public override bool RequiresWin => false;

	public override bool MultiplayerOnly => true;

	public TeamPlayer(SerializableRun run, ulong playerId)
		: base(run, playerId)
	{
	}

	public override bool IsObtained()
	{
		int num = 0;
		foreach (SerializableCard item in _localPlayer.Deck)
		{
			if (SaveUtil.CardOrDeprecated(item.Id).MultiplayerConstraint == CardMultiplayerConstraint.MultiplayerOnly)
			{
				num++;
			}
		}
		return num >= 3;
	}
}
