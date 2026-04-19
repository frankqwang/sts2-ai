using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Models.Badges;

public class BigDeck : Badge
{
	public override string Id => "BIG_DECK";

	public override BadgeRarity Rarity
	{
		get
		{
			int count = _localPlayer.Deck.Count;
			if (count >= 60)
			{
				if (count >= 100)
				{
					return BadgeRarity.Gold;
				}
				return BadgeRarity.Silver;
			}
			if (count >= 40)
			{
				return BadgeRarity.Bronze;
			}
			return BadgeRarity.None;
		}
	}

	public override bool RequiresWin => true;

	public override bool MultiplayerOnly => false;

	public BigDeck(SerializableRun run, ulong playerId)
		: base(run, playerId)
	{
	}

	public override bool IsObtained()
	{
		return Rarity != BadgeRarity.None;
	}
}
