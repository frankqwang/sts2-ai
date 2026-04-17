using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Models.Badges;

public class TabletBadge : Badge
{
	public override string Id => "TABLET";

	public override BadgeRarity Rarity => BadgeRarity.Gold;

	public override bool RequiresWin => true;

	public override bool MultiplayerOnly => false;

	public TabletBadge(SerializableRun run, ulong playerId)
		: base(run, playerId)
	{
	}

	public override bool IsObtained()
	{
		return _localPlayer.MaxHp == 1;
	}
}
