using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Saves.Runs;

namespace MegaCrit.Sts2.Core.Models.Badges;

public class Curses : Badge
{
	private const int _curseRequirement = 5;

	public override string Id => "CURSES";

	public override BadgeRarity Rarity => BadgeRarity.Bronze;

	public override bool RequiresWin => true;

	public override bool MultiplayerOnly => false;

	public Curses(SerializableRun run, ulong playerId)
		: base(run, playerId)
	{
	}

	public override bool IsObtained()
	{
		int num = 0;
		foreach (SerializableCard item in _localPlayer.Deck)
		{
			if (SaveUtil.CardOrDeprecated(item.Id).Type == CardType.Curse)
			{
				num++;
			}
		}
		return num >= 5;
	}
}
