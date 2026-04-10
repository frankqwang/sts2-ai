using System;
using System.Globalization;
using System.Linq;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Simulation;

namespace MegaCrit.Sts2.Core.Training;

public static class CombatTrainingMode
{
	static CombatTrainingMode()
	{
		Func<bool> existingCheck = NonInteractiveMode.AutoSlayerCheck;
		NonInteractiveMode.AutoSlayerCheck = () => existingCheck() || IsActive || CombatSimulationMode.IsAnyActive || FullRunSimulationMode.IsAnyActive;
	}

	public static bool IsActive => CommandLineHelper.HasArg("combat-trainer");

	public static bool ShouldLoop => IsActive && CommandLineHelper.HasArg("trainer-loop");

	public static bool ShouldQuitOnCombatEnd => IsActive && CommandLineHelper.HasArg("trainer-quit-on-end");

	public static string ResolveEpisodeSeed(string? seedOverride = null)
	{
		string? seed = string.IsNullOrWhiteSpace(seedOverride) ? CommandLineHelper.GetValue("trainer-seed") : seedOverride;
		if (!string.IsNullOrWhiteSpace(seed))
		{
			return seed;
		}
		return SeedHelper.GetRandomSeed();
	}

	public static int ResolveAscensionLevel(int? ascensionLevelOverride = null)
	{
		if (ascensionLevelOverride.HasValue)
		{
			return Math.Max(ascensionLevelOverride.Value, 0);
		}
		string? value = CommandLineHelper.GetValue("trainer-ascension");
		if (string.IsNullOrWhiteSpace(value))
		{
			return 0;
		}
		if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int result))
		{
			throw new InvalidOperationException($"Invalid --trainer-ascension value: '{value}'.");
		}
		return Math.Max(result, 0);
	}

	public static CharacterModel ResolveCharacter(string? characterIdOverride = null)
	{
		string? value = string.IsNullOrWhiteSpace(characterIdOverride) ? CommandLineHelper.GetValue("trainer-character") : characterIdOverride;
		if (string.IsNullOrWhiteSpace(value))
		{
			return ModelDb.AllCharacters.First();
		}
		ModelId modelId = new ModelId(ModelId.SlugifyCategory<CharacterModel>(), value.ToUpperInvariant());
		return ModelDb.GetById<CharacterModel>(modelId);
	}

	public static EncounterModel ResolveEncounter(string? encounterIdOverride = null)
	{
		string? value = string.IsNullOrWhiteSpace(encounterIdOverride) ? CommandLineHelper.GetValue("trainer-encounter") : encounterIdOverride;
		if (string.IsNullOrWhiteSpace(value))
		{
			return ModelDb.AllEncounters.Where(static encounter => encounter.RoomType == RoomType.Monster).OrderBy(static encounter => encounter.Id.Entry).First();
		}
		ModelId modelId = new ModelId(ModelId.SlugifyCategory<EncounterModel>(), value.ToUpperInvariant());
		return ModelDb.GetById<EncounterModel>(modelId);
	}
}
