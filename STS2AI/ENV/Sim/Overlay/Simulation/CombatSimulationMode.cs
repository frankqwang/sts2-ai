using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.Simulation;

public static class CombatSimulationMode
{
	public static bool IsServerActive => CommandLineHelper.HasArg("combat-sim-server");

	public static bool IsSmokeActive => CommandLineHelper.HasArg("combat-sim-smoke");

	public static bool IsBenchmarkActive => CommandLineHelper.HasArg("combat-sim-benchmark");

	public static bool IsAnyActive => IsServerActive || IsSmokeActive || IsBenchmarkActive;

	public static int StepBudget
	{
		get
		{
			string? value = CommandLineHelper.GetValue("combat-sim-steps");
			if (!int.TryParse(value, out int result) || result <= 0)
			{
				return 32;
			}
			return result;
		}
	}

	public static int EpisodeCount
	{
		get
		{
			string? value = CommandLineHelper.GetValue("combat-sim-episodes");
			if (!int.TryParse(value, out int result) || result <= 0)
			{
				return 10;
			}
			return result;
		}
	}

	public static string? CharacterId => CommandLineHelper.GetValue("combat-sim-character");

	public static string? EncounterId => CommandLineHelper.GetValue("combat-sim-encounter");

	public static string? Seed => CommandLineHelper.GetValue("combat-sim-seed");

	public static int? AscensionLevel
	{
		get
		{
			string? value = CommandLineHelper.GetValue("combat-sim-ascension");
			if (!int.TryParse(value, out int result))
			{
				return null;
			}
			return result;
		}
	}
}
