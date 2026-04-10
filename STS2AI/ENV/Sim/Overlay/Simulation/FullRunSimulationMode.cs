using MegaCrit.Sts2.Core.Helpers;
using System;

namespace MegaCrit.Sts2.Core.Simulation;

public static class FullRunSimulationMode
{
	public static bool IsServerActive => CommandLineHelper.HasArg("full-run-sim-server");

	public static bool IsSmokeActive => CommandLineHelper.HasArg("full-run-sim-smoke");

	public static bool IsAnyActive => IsServerActive || IsSmokeActive;

	public static string? CharacterId => CommandLineHelper.GetValue("full-run-sim-character");

	public static string? Seed => CommandLineHelper.GetValue("full-run-sim-seed");

	public static int? AscensionLevel
	{
		get
		{
			string? value = CommandLineHelper.GetValue("full-run-sim-ascension");
			if (!int.TryParse(value, out int result))
			{
				return null;
			}
			return result;
		}
	}

	public static int? ProfileId
	{
		get
		{
			string? value = CommandLineHelper.GetValue("full-run-sim-profile-id");
			if (!int.TryParse(value, out int result))
			{
				return null;
			}
			return Math.Max(1, result);
		}
	}
}
