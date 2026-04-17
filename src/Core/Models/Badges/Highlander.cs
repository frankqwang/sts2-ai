using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Saves.Runs;

namespace MegaCrit.Sts2.Core.Models.Badges;

public class Highlander : Badge
{
	public override string Id => "HIGHLANDER";

	public override BadgeRarity Rarity => BadgeRarity.Bronze;

	public override bool RequiresWin => true;

	public override bool MultiplayerOnly => false;

	public Highlander(SerializableRun run, ulong playerId)
		: base(run, playerId)
	{
	}

	public override bool IsObtained()
	{
		List<SerializableCard> list = _localPlayer.Deck.Where((SerializableCard c) => SaveUtil.CardOrDeprecated(c.Id).Rarity != CardRarity.Basic).ToList();
		return list.Select((SerializableCard card) => card.Id).Distinct().Count() == list.Count;
	}
}
