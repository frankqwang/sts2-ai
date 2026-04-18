using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Models.Badges;

public class Famished : Badge
{
	public override string Id => "FAMISHED";

	public override BadgeRarity Rarity => BadgeRarity.Bronze;

	public override bool RequiresWin => true;

	public override bool MultiplayerOnly => false;

	public Famished(SerializableRun run, ulong playerId)
		: base(run, playerId)
	{
	}

	public override bool IsObtained()
	{
		int num = (SaveUtil.CharacterOrDeprecated(_localPlayer.CharacterId).StartingHp + 1) / 2;
		return _localPlayer.MaxHp < num;
	}
}
