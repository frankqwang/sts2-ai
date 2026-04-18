using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Training;

namespace MegaCrit.Sts2.Core.Simulation;

public sealed class CombatSimulatorBenchmarkRunner
{
	private static readonly string ArtifactDir = Path.Combine("artifacts", "combat_sim_benchmark");

	private static readonly string SummaryPath = Path.Combine(ArtifactDir, "latest_summary.json");

	public async Task<int> StartAsync(NGame game)
	{
		if (game == null)
		{
			throw new ArgumentNullException(nameof(game));
		}

		Directory.CreateDirectory(ArtifactDir);
		var benchmark = new CombatSimulatorBenchmarkResult
		{
			character_id = CombatSimulationMode.CharacterId,
			encounter_id = CombatSimulationMode.EncounterId,
			ascension_level = CombatSimulationMode.AscensionLevel ?? 0,
			episodes_requested = CombatSimulationMode.EpisodeCount,
			step_budget = CombatSimulationMode.StepBudget,
			episodes = new List<CombatSimulatorEpisodeResult>()
		};

		Stopwatch totalStopwatch = Stopwatch.StartNew();
		try
		{
			Log.Info($"[CombatSimBenchmark] Starting pure simulator benchmark. episodes={CombatSimulationMode.EpisodeCount} steps={CombatSimulationMode.StepBudget}");
			for (int episodeIndex = 0; episodeIndex < CombatSimulationMode.EpisodeCount; episodeIndex++)
			{
				CombatSimulatorEpisodeResult episode = await RunEpisodeAsync(episodeIndex + 1);
				benchmark.episodes.Add(episode);
				Console.WriteLine(JsonSerializer.Serialize(episode));
				if (!episode.completed)
				{
					benchmark.completed = false;
					benchmark.failure = $"episode {episode.episode_index} did not complete";
					benchmark.total_wall_duration_s = Math.Round(totalStopwatch.Elapsed.TotalSeconds, 4);
					ComputeSummary(benchmark);
					WriteSummary(benchmark);
					Log.Error($"[CombatSimBenchmark] Episode {episode.episode_index} failed: {episode.failure ?? "unknown"}");
					return 2;
				}
			}

			benchmark.completed = true;
			benchmark.total_wall_duration_s = Math.Round(totalStopwatch.Elapsed.TotalSeconds, 4);
			ComputeSummary(benchmark);
			WriteSummary(benchmark);
			Console.WriteLine(JsonSerializer.Serialize(benchmark));
			Log.Info($"[CombatSimBenchmark] Finished. episodes={benchmark.episodes_completed} wall_s={benchmark.total_wall_duration_s} core_s={benchmark.total_core_duration_s}");
			return 0;
		}
		catch (Exception ex)
		{
			benchmark.completed = false;
			benchmark.failure = $"{ex.GetType().Name}: {ex.Message}";
			benchmark.total_wall_duration_s = Math.Round(totalStopwatch.Elapsed.TotalSeconds, 4);
			ComputeSummary(benchmark);
			WriteSummary(benchmark);
			Log.Error($"[CombatSimBenchmark] Failed: {ex}");
			return 1;
		}
	}

	private static async Task<CombatSimulatorEpisodeResult> RunEpisodeAsync(int episodeIndex)
	{
		Stopwatch resetStopwatch = Stopwatch.StartNew();
		CombatTrainingStateSnapshot state = await CombatTrainingEnvService.Instance.ResetAsync(new CombatTrainingResetRequest
		{
			CharacterId = CombatSimulationMode.CharacterId,
			EncounterId = CombatSimulationMode.EncounterId,
			Seed = CombatSimulationMode.Seed,
			AscensionLevel = CombatSimulationMode.AscensionLevel
		});
		resetStopwatch.Stop();

		Stopwatch episodeStopwatch = Stopwatch.StartNew();
		int steps = 0;
		string? failure = null;

		while (!state.IsEpisodeDone && steps < CombatSimulationMode.StepBudget)
		{
			CombatTrainingActionRequest action = CombatSimulatorAutoplay.BuildAction(state);
			CombatTrainingStepResult result = await CombatTrainingEnvService.Instance.StepAsync(action);
			if (!result.Accepted)
			{
				failure = result.Error ?? "Action rejected";
				break;
			}
			if (result.State == null)
			{
				failure = "Missing state snapshot";
				break;
			}
			state = result.State;
			steps++;
		}

		episodeStopwatch.Stop();
		if (!state.IsEpisodeDone && failure == null && steps >= CombatSimulationMode.StepBudget)
		{
			failure = $"Step budget exhausted at {steps}";
		}

		return new CombatSimulatorEpisodeResult
		{
			episode_index = episodeIndex,
			episode_number = state.EpisodeNumber,
			seed = state.Seed,
			character_id = state.CharacterId,
			encounter_id = state.EncounterId,
			ascension_level = state.AscensionLevel,
			steps = steps,
			reset_duration_s = Math.Round(resetStopwatch.Elapsed.TotalSeconds, 4),
			combat_duration_s = Math.Round(episodeStopwatch.Elapsed.TotalSeconds, 4),
			episode_duration_s = Math.Round(resetStopwatch.Elapsed.TotalSeconds + episodeStopwatch.Elapsed.TotalSeconds, 4),
			steps_per_second = steps > 0 && episodeStopwatch.Elapsed.TotalSeconds > 0 ? Math.Round(steps / episodeStopwatch.Elapsed.TotalSeconds, 4) : 0.0,
			completed = state.IsEpisodeDone && failure == null,
			victory = state.Victory,
			failure = failure
		};
	}

	private static void ComputeSummary(CombatSimulatorBenchmarkResult benchmark)
	{
		benchmark.episodes_completed = benchmark.episodes.Count;
		double totalReset = 0.0;
		double totalCombat = 0.0;
		int totalSteps = 0;
		int victories = 0;
		int defeats = 0;
		foreach (CombatSimulatorEpisodeResult episode in benchmark.episodes)
		{
			totalReset += episode.reset_duration_s;
			totalCombat += episode.combat_duration_s;
			totalSteps += episode.steps;
			if (episode.victory == true)
			{
				victories++;
			}
			else if (episode.victory == false)
			{
				defeats++;
			}
		}

		benchmark.total_steps = totalSteps;
		benchmark.total_reset_duration_s = Math.Round(totalReset, 4);
		benchmark.total_core_duration_s = Math.Round(totalReset + totalCombat, 4);
		benchmark.avg_reset_duration_s = benchmark.episodes.Count > 0 ? Math.Round(totalReset / benchmark.episodes.Count, 4) : 0.0;
		benchmark.avg_combat_duration_s = benchmark.episodes.Count > 0 ? Math.Round(totalCombat / benchmark.episodes.Count, 4) : 0.0;
		benchmark.avg_episode_duration_s = benchmark.episodes.Count > 0 ? Math.Round((totalReset + totalCombat) / benchmark.episodes.Count, 4) : 0.0;
		benchmark.aggregate_steps_per_second = totalCombat > 0 ? Math.Round(totalSteps / totalCombat, 4) : 0.0;
		benchmark.aggregate_episodes_per_second = benchmark.total_wall_duration_s > 0 ? Math.Round(benchmark.episodes.Count / benchmark.total_wall_duration_s, 4) : 0.0;
		benchmark.victories = victories;
		benchmark.defeats = defeats;
	}

	private static void WriteSummary(CombatSimulatorBenchmarkResult benchmark)
	{
		string json = JsonSerializer.Serialize(benchmark, new JsonSerializerOptions
		{
			WriteIndented = true
		});
		File.WriteAllText(SummaryPath, json);
	}

	private sealed class CombatSimulatorEpisodeResult
	{
		public int episode_index { get; set; }

		public int episode_number { get; set; }

		public string? seed { get; set; }

		public string? character_id { get; set; }

		public string? encounter_id { get; set; }

		public int ascension_level { get; set; }

		public int steps { get; set; }

		public double reset_duration_s { get; set; }

		public double combat_duration_s { get; set; }

		public double episode_duration_s { get; set; }

		public double steps_per_second { get; set; }

		public bool completed { get; set; }

		public bool? victory { get; set; }

		public string? failure { get; set; }
	}

	private sealed class CombatSimulatorBenchmarkResult
	{
		public bool completed { get; set; }

		public string? failure { get; set; }

		public string? character_id { get; set; }

		public string? encounter_id { get; set; }

		public int ascension_level { get; set; }

		public int episodes_requested { get; set; }

		public int episodes_completed { get; set; }

		public int step_budget { get; set; }

		public int total_steps { get; set; }

		public double total_reset_duration_s { get; set; }

		public double total_core_duration_s { get; set; }

		public double total_wall_duration_s { get; set; }

		public double avg_reset_duration_s { get; set; }

		public double avg_combat_duration_s { get; set; }

		public double avg_episode_duration_s { get; set; }

		public double aggregate_steps_per_second { get; set; }

		public double aggregate_episodes_per_second { get; set; }

		public int victories { get; set; }

		public int defeats { get; set; }

		public List<CombatSimulatorEpisodeResult> episodes { get; set; } = new List<CombatSimulatorEpisodeResult>();
	}
}
