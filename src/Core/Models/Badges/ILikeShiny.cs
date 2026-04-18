using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Models.Badges;

public class ILikeShiny : Badge
{
	private const int _relicRequirement = 25;

	public override string Id => "ILIKESHINY";

	public override BadgeRarity Rarity => BadgeRarity.Bronze;

	public override bool RequiresWin => false;

	public override bool MultiplayerOnly => false;

	public ILikeShiny(SerializableRun run, ulong playerId)
		: base(run, playerId)
	{
	}

	public override bool IsObtained()
	{
		return _localPlayer.Relics.Count >= 25;
	}
}
