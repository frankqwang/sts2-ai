using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Simulation;
using MegaCrit.Sts2.Core.TestSupport;

namespace HeadlessSim;

internal enum HostProtocol
{
	Proto,
}

// Entry point + host bootstrap. Request processing is split across partial
// files by concern:
//
//   Program.Pipe.cs       named-pipe server and read/write framing
//   Program.Proto.cs      protobuf request router + every ProcessProto* handler
//   Program.StepSupport.cs full-run step decision-state advancement helper
//
// All partials are the same `internal static partial class Program`, so private
// members (JsonOptions / RequestStateCache / ORT fields / helpers) are shared
// without any cross-file access tweaks.
internal static partial class Program
{
	private sealed class RequestStateCache
	{
		public FullRunSimulationStateSnapshot? Snapshot { get; set; }

		public FullRunApiState? ApiState { get; set; }
	}

	private static readonly JsonSerializerOptions JsonOptions = new JsonSerializerOptions
	{
		PropertyNameCaseInsensitive = true,
		DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
	};

	public static async Task Main(string[] args)
	{
		// ThreadPool min threads: configurable via env var for A/B testing.
		// Default .NET min is often CPU core count, causing Task.Yield()
		// continuation delays when multiple HeadlessSim processes compete.
		string? minThreadsEnv = Environment.GetEnvironmentVariable("STS2_MIN_THREADS");
		int minThreads = 32; // default boost
		if (!string.IsNullOrEmpty(minThreadsEnv) && int.TryParse(minThreadsEnv, out int parsed) && parsed > 0)
			minThreads = parsed;
		System.Threading.ThreadPool.SetMinThreads(minThreads, minThreads);
		{
			int mw, mio;
			System.Threading.ThreadPool.GetMinThreads(out mw, out mio);
			Console.Error.WriteLine($"[THREADPOOL] MinThreads={mw} (requested={minThreads})");
		}

		HostOptions options = HostOptions.Parse(args);
		BootstrapStandaloneRuntime();
		if (options.ExportCardRuntimeTextsPath != null)
		{
			await ExportCardRuntimeTextsAsync(options.ExportCardRuntimeTextsPath, options.ExportLocales);
			return;
		}
		using IDisposable standaloneScope = FullRunTrainingEnvService.EnterStandaloneMode();
		FullRunTrainingEnvService service = FullRunTrainingEnvService.Instance;

		Console.Error.WriteLine($"HeadlessSim: pipe mode ready on \\\\.\\pipe\\{options.PipeName}");
		await RunPipeServerAsync(service, options);
	}

	private static void BootstrapStandaloneRuntime()
	{
		TestMode.IsOn = true;
		UserDataPathProvider.IsRunningModded = false;
		SaveManager saveManager = SaveManager.Instance;
		saveManager.InitSettingsDataForTest();
		EnsureLocalizationInitialized();
		ModelDb.Init();
		ModelIdSerializationCache.Init();
		ModelDb.InitIds();
		saveManager.InitProfileId(profileId: 1);
		saveManager.InitProgressData();
		saveManager.InitPrefsDataForTest();
	}

	private static void EnsureLocalizationInitialized()
	{
		if (LocManager.Instance == null)
		{
			LocManager.Initialize();
		}

		string locale = Environment.GetEnvironmentVariable("STS2_HEADLESS_LOCALE")?.Trim().ToLowerInvariant() ?? "";
		if (string.IsNullOrWhiteSpace(locale))
		{
			locale = "zhs";
		}
		LocManager.Instance.SetLanguage(locale);
	}

	private static async Task ExportCardRuntimeTextsAsync(string outputPath, IReadOnlyList<string> locales)
	{
		List<CardRuntimeTextRecord> rows = new List<CardRuntimeTextRecord>();
		List<CardModel> cards = ModelDb.AllCards.OrderBy((CardModel c) => c.Id.Entry, StringComparer.OrdinalIgnoreCase).ToList();
		foreach (string locale in locales)
		{
			LocManager.Instance.SetLanguage(locale);
			foreach (CardModel card in cards)
			{
				rows.Add(new CardRuntimeTextRecord
				{
					Id = card.Id.Entry.ToLowerInvariant(),
					ClassName = card.GetType().Name,
					Locale = locale,
					Title = card.Title,
					DescriptionRuntime = card.GetDescriptionForPile(PileType.None),
					UpgradePreviewRuntime = card.GetDescriptionForUpgradePreview()
				});
			}
		}

		string? directory = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(directory))
		{
			Directory.CreateDirectory(directory);
		}

		JsonSerializerOptions exportOptions = new JsonSerializerOptions(JsonOptions)
		{
			WriteIndented = true
		};
		await File.WriteAllTextAsync(outputPath, JsonSerializer.Serialize(rows, exportOptions), Encoding.UTF8);
		Console.Error.WriteLine($"HeadlessSim: exported runtime card texts -> {outputPath} ({rows.Count} rows)");
	}

	private sealed class CardRuntimeTextRecord
	{
		public string Id { get; set; } = string.Empty;

		public string ClassName { get; set; } = string.Empty;

		public string Locale { get; set; } = string.Empty;

		public string Title { get; set; } = string.Empty;

		public string DescriptionRuntime { get; set; } = string.Empty;

		public string UpgradePreviewRuntime { get; set; } = string.Empty;
	}

	// Maps exception type / ErrorCode property to a structured error code string.
	// Used by the pipe transport when wrapping handler exceptions.
	private static string? GetStructuredErrorCode(Exception exception)
	{
		if (exception is JsonException)
		{
			return "invalid_json";
		}

		PropertyInfo? errorCodeProperty = exception.GetType().GetProperty(
			"ErrorCode",
			BindingFlags.Public | BindingFlags.Instance);
		if (errorCodeProperty?.PropertyType == typeof(string))
		{
			return errorCodeProperty.GetValue(exception) as string;
		}

		return exception switch
		{
			InvalidOperationException => "invalid_request",
			TimeoutException => "request_timeout",
			_ => null
		};
	}

	private sealed class HostOptions
	{
		public int Port { get; private set; } = 15527;

		public HostProtocol Protocol { get; private set; } = HostProtocol.Proto;

		public TimeSpan ReadTimeout { get; private set; } = TimeSpan.FromSeconds(60);

		public TimeSpan RequestTimeout { get; private set; } = TimeSpan.FromSeconds(45);

		public string? ExportCardRuntimeTextsPath { get; private set; }

		public IReadOnlyList<string> ExportLocales { get; private set; } = new[] { "eng", "zhs" };

		public string PipeName => $"sts2_mcts_proto_{Port}";

		public static HostOptions Parse(IEnumerable<string> args)
		{
			HostOptions options = new HostOptions();
			string[] values = args.ToArray();
			for (int i = 0; i < values.Length; i++)
			{
				switch (values[i])
				{
					case "--port" when i + 1 < values.Length && int.TryParse(values[i + 1], out int port):
						options.Port = port;
						i++;
						break;
					case "--read-timeout-seconds" when i + 1 < values.Length && double.TryParse(values[i + 1], out double readSeconds):
						options.ReadTimeout = TimeSpan.FromSeconds(Math.Max(1, readSeconds));
						i++;
						break;
					case "--request-timeout-seconds" when i + 1 < values.Length && double.TryParse(values[i + 1], out double requestSeconds):
						options.RequestTimeout = TimeSpan.FromSeconds(Math.Max(1, requestSeconds));
						i++;
						break;
					case "--protocol" when i + 1 < values.Length:
						string protocol = values[i + 1].Trim().ToLowerInvariant();
						options.Protocol = protocol switch
						{
							"json" => throw new InvalidOperationException(
								"--protocol json 已下线。HeadlessSim 只支持 protobuf pipe，请用 --protocol proto。"),
							"bin" or "binary" => throw new InvalidOperationException(
								"--protocol binary (手写二进制 wire) 已废弃。请用 --protocol proto。"),
							"proto" or "protobuf" => HostProtocol.Proto,
							_ => throw new InvalidOperationException($"Unknown protocol '{values[i + 1]}'. Expected 'proto'.")
						};
						i++;
						break;
					case "--export-card-runtime-texts" when i + 1 < values.Length:
						options.ExportCardRuntimeTextsPath = values[i + 1];
						i++;
						break;
					case "--export-locales" when i + 1 < values.Length:
						options.ExportLocales = values[i + 1]
							.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
							.Select(static s => s.ToLowerInvariant())
							.ToArray();
						i++;
						break;
				}
			}

			return options;
		}
	}
}
