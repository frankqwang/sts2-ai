using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Relics;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Localization.DynamicVars;
using MegaCrit.Sts2.Core.Models.CardPools;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Models.Relics;

public sealed class NeowsBones : RelicModel
{
	private const string _relicCountKey = "Relics";

	private const string _cursesCountKey = "Curses";

	public override RelicRarity Rarity => RelicRarity.Ancient;

	protected override IEnumerable<DynamicVar> CanonicalVars => new global::_003C_003Ez__ReadOnlyArray<DynamicVar>(new DynamicVar[2]
	{
		new DynamicVar("Relics", 2m),
		new DynamicVar("Curses", 1m)
	});

	public static IEnumerable<RelicModel> GetValidRelics(IRunState state)
	{
		return (from o in ModelDb.Event<Neow>().AllPossibleOptions
			where o.Relic != null && o.Relic.IsAllowed(state) && !(o.Relic is NeowsBones)
			select o.Relic).OfType<RelicModel>();
	}

	public override async Task AfterObtained()
	{
		List<RelicModel> list = GetValidRelics(base.Owner.RunState).ToList();
		base.Owner.PlayerRng.Rewards.Shuffle(list);
		List<Reward> rewards = list.Take(base.DynamicVars["Relics"].IntValue).Select((Func<RelicModel, Reward>)((RelicModel r) => new RelicReward(r, base.Owner))).ToList();
		await new RewardsSet(base.Owner).WithCustomRewards(rewards).WithSkippingDisallowed().Offer();
		if (LocalContext.IsMe(base.Owner))
		{
			HashSet<CardModel> availableCurses = (from c in ModelDb.CardPool<CurseCardPool>().GetUnlockedCards(base.Owner.UnlockState, base.Owner.RunState.CardMultiplayerConstraint)
				where c.CanBeGeneratedByModifiers
				select c).ToHashSet();
			List<CardPileAddResult> curseResults = new List<CardPileAddResult>();
			for (int i = 0; i < base.DynamicVars["Curses"].IntValue; i++)
			{
				CardModel cardModel = base.Owner.RunState.Rng.Niche.NextItem(availableCurses);
				availableCurses.Remove(cardModel);
				CardModel card = base.Owner.RunState.CreateCard(cardModel, base.Owner);
				CardPileAddResult item = await CardPileCmd.Add(card, PileType.Deck);
				curseResults.Add(item);
				RunManager.Instance.RewardSynchronizer.SyncLocalObtainedCard(item.cardAdded);
			}
			CardCmd.PreviewCardPileAdd(curseResults, 2f);
			await Cmd.Wait(0.75f);
		}
		else
		{
			for (int num = 0; num < base.DynamicVars["Curses"].IntValue; num++)
			{
				base.Owner.RunState.Rng.Niche.NextInt();
			}
		}
	}
}
