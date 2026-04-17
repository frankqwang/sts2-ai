using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Models.Badges;

public class TinyDeck : Badge
{
	public override string Id => "TINY_DECK";

	public override BadgeRarity Rarity
	{
		get
		{
			int count = _localPlayer.Deck.Count;
			if (count <= 10)
			{
				if (count <= 5)
				{
					return BadgeRarity.Gold;
				}
				return BadgeRarity.Silver;
			}
			if (count <= 20)
			{
				return BadgeRarity.Bronze;
			}
			return BadgeRarity.None;
		}
	}

	public override bool RequiresWin => true;

	public override bool MultiplayerOnly => false;

	public TinyDeck(SerializableRun run, ulong playerId)
		: base(run, playerId)
	{
	}

	public override bool IsObtained()
	{
		return Rarity != BadgeRarity.None;
	}
}
