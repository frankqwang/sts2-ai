using System.Collections.Generic;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Models.Badges;

public static class BadgePool
{
	public static IReadOnlyCollection<Badge> CreateAll(SerializableRun run, ulong playerId)
	{
		return new global::_003C_003Ez__ReadOnlyArray<Badge>(new Badge[20]
		{
			new Curses(run, playerId),
			new DoubleSnecko(run, playerId),
			new EliteKiller(run, playerId),
			new Famished(run, playerId),
			new Glutton(run, playerId),
			new Healer(run, playerId),
			new Highlander(run, playerId),
			new Honed(run, playerId),
			new BigDeck(run, playerId),
			new ILikeShiny(run, playerId),
			new KaChing(run, playerId),
			new MoneyMoney(run, playerId),
			new MysteryMachine(run, playerId),
			new Perfect(run, playerId),
			new Restful(run, playerId),
			new Restless(run, playerId),
			new Speedy(run, playerId),
			new TabletBadge(run, playerId),
			new TeamPlayer(run, playerId),
			new TinyDeck(run, playerId)
		});
	}
}
