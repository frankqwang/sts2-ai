using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Models.Badges;

public class Debuffer : Badge
{
	public override string Id => "DEBUFFER";

	public override BadgeRarity Rarity => BadgeRarity.Bronze;

	public override bool RequiresWin => false;

	public override bool MultiplayerOnly => true;

	public Debuffer(SerializableRun run, ulong playerId)
		: base(run, playerId)
	{
	}

	public override bool IsObtained()
	{
		return false;
	}
}
