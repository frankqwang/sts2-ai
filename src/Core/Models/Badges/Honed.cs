using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Saves.Runs;

namespace MegaCrit.Sts2.Core.Models.Badges;

public class Honed : Badge
{
	private const int _sameCardAmount = 5;

	public override string Id => "HONED";

	public override BadgeRarity Rarity => BadgeRarity.Bronze;

	public override bool RequiresWin => true;

	public override bool MultiplayerOnly => false;

	public Honed(SerializableRun run, ulong playerId)
		: base(run, playerId)
	{
	}

	public override bool IsObtained()
	{
		List<SerializableCard> source = _localPlayer.Deck.Where((SerializableCard c) => SaveUtil.CardOrDeprecated(c.Id).Rarity != CardRarity.Basic).ToList();
		return (from c in source
			group c by c.Id).Any((IGrouping<ModelId, SerializableCard> g) => g.Count() >= 5);
	}
}
