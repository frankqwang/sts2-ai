using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Localization.DynamicVars;
using MegaCrit.Sts2.Core.ValueProps;

namespace MegaCrit.Sts2.Core.Models.Cards;

public sealed class FollowThrough : CardModel
{
	private const string _cardCountKey = "CardCount";

	protected override bool ShouldGlowGoldInternal => IsPlayedAnAdditionalTime;

	private bool IsPlayedAnAdditionalTime
	{
		get
		{
			decimal? num = base.Owner.PlayerCombatState?.Hand.Cards.Count((CardModel c) => c != this);
			decimal baseValue = base.DynamicVars["CardCount"].BaseValue;
			return (num.GetValueOrDefault() >= baseValue) & num.HasValue;
		}
	}

	protected override IEnumerable<DynamicVar> CanonicalVars => new global::_003C_003Ez__ReadOnlyArray<DynamicVar>(new DynamicVar[2]
	{
		new DamageVar(7m, ValueProp.Move),
		new DynamicVar("CardCount", 5m)
	});

	public FollowThrough()
		: base(1, CardType.Attack, CardRarity.Common, TargetType.AnyEnemy)
	{
	}

	protected override async Task OnPlay(PlayerChoiceContext choiceContext, CardPlay cardPlay)
	{
		ArgumentNullException.ThrowIfNull(cardPlay.Target, "cardPlay.Target");
		await DamageCmd.Attack(base.DynamicVars.Damage.BaseValue).FromCard(this).Targeting(cardPlay.Target)
			.WithHitCount((!IsPlayedAnAdditionalTime) ? 1 : 2)
			.WithHitFx("vfx/vfx_attack_slash")
			.Execute(choiceContext);
	}

	protected override void OnUpgrade()
	{
		base.DynamicVars.Damage.UpgradeValueBy(2m);
	}
}
