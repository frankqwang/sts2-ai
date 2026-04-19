using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Models.Badges;

public class FavoriteCard : Badge
{
	private const int _cardPlayRequirement = 100;

	public override string Id => "FAVORITE_CARD";

	public override BadgeRarity Rarity => BadgeRarity.Bronze;

	public override bool RequiresWin => true;

	public override bool MultiplayerOnly => false;

	public FavoriteCard(SerializableRun run, ulong playerId)
		: base(run, playerId)
	{
	}

	public override bool IsObtained()
	{
		return false;
	}
}
