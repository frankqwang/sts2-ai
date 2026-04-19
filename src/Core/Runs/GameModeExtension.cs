namespace MegaCrit.Sts2.Core.Runs;

public static class GameModeExtension
{
	public static bool AreAchievementsAndEpochsLocked(this GameMode gameMode)
	{
		return gameMode != GameMode.Standard;
	}
}
