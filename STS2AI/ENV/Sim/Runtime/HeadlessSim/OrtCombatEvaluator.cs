using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MegaCrit.Sts2.Core.Simulation;

namespace HeadlessSim;

internal sealed class CombatEvaluationResult
{
	public float[] PolicyLogits { get; init; } = Array.Empty<float>();

	public float Value { get; init; }
}

internal sealed class OrtCombatEvaluator : IDisposable
{
	private enum OrtDevicePreference
	{
		Auto,
		Cpu,
		Cuda,
	}

	private sealed class SessionBootstrapResult
	{
		public InferenceSession Session { get; init; } = null!;

		public string RequestedDevice { get; init; } = "auto";

		public string ExecutionProviderName { get; init; } = "CPUExecutionProvider";

		public bool FellBackToCpu { get; init; }

		public string? FallbackReason { get; init; }
	}

	private static int _debugDumpWritten;
	private static readonly object _cudaDependencyPathLock = new();
	private static bool _cudaDependencyPathPrepared;
	private readonly InferenceSession _session;
	private readonly CombatFeatureEncoder _encoder;

	public CombatModelMetadata Metadata { get; }

	public CombatVocab Vocab { get; }

	public string RequestedDevice { get; }

	public string ExecutionProviderName { get; }

	public bool FellBackToCpu { get; }

	public OrtCombatEvaluator(string onnxPath, string? vocabPath = null)
	{
		SessionBootstrapResult bootstrap = CreateSession(onnxPath);
		_session = bootstrap.Session;
		Metadata = InspectModel(_session);
		Vocab = CombatVocab.Load(onnxPath, vocabPath);
		_encoder = new CombatFeatureEncoder(Vocab);
		RequestedDevice = bootstrap.RequestedDevice;
		ExecutionProviderName = bootstrap.ExecutionProviderName;
		FellBackToCpu = bootstrap.FellBackToCpu;

		if (!string.IsNullOrWhiteSpace(bootstrap.FallbackReason))
		{
			Console.Error.WriteLine($"[ORT] CUDA unavailable; falling back to CPU: {bootstrap.FallbackReason}");
		}

		Console.Error.WriteLine(
			$"[ORT] Execution provider={ExecutionProviderName} requested={RequestedDevice} fallback={(FellBackToCpu ? "yes" : "no")}");
	}

	public CombatEvaluationResult Evaluate(
		FullRunSimulationStateSnapshot snapshot,
		IReadOnlyList<FullRunSimulationLegalAction> legalActions,
		bool useContinuationValue)
	{
		return EvaluateBatch(new[] { snapshot }, new[] { legalActions }, useContinuationValue)[0];
	}

	public IReadOnlyList<CombatEvaluationResult> EvaluateBatch(
		IReadOnlyList<FullRunSimulationStateSnapshot> snapshots,
		IReadOnlyList<IReadOnlyList<FullRunSimulationLegalAction>> legalActionsBatch,
		bool useContinuationValue)
	{
		if (snapshots.Count != legalActionsBatch.Count)
		{
			throw new InvalidOperationException("State batch and legal-action batch length mismatch.");
		}

		if (snapshots.Count == 0)
		{
			return Array.Empty<CombatEvaluationResult>();
		}

		List<CombatEncodedFeatures> encoded = new(snapshots.Count);
		for (int i = 0; i < snapshots.Count; i++)
		{
			encoded.Add(_encoder.Encode(snapshots[i], legalActionsBatch[i], Metadata));
		}

		List<NamedOnnxValue> inputs = BuildInputs(encoded);
		using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);

		Tensor<float> logitsTensor = results.First(result =>
			string.Equals(result.Name, Metadata.PolicyOutputName, StringComparison.OrdinalIgnoreCase)).AsTensor<float>();

		Tensor<float>? valueTensor = Metadata.HasValueOutput && !string.IsNullOrWhiteSpace(Metadata.ValueOutputName)
			? results.First(result => string.Equals(result.Name, Metadata.ValueOutputName, StringComparison.OrdinalIgnoreCase)).AsTensor<float>()
			: null;

		Tensor<float>? continuationTensor = Metadata.HasContinuationOutput && !string.IsNullOrWhiteSpace(Metadata.ContinuationOutputName)
			? results.First(result => string.Equals(result.Name, Metadata.ContinuationOutputName, StringComparison.OrdinalIgnoreCase)).AsTensor<float>()
			: null;

		CombatEvaluationResult[] output = new CombatEvaluationResult[snapshots.Count];
		for (int batchIndex = 0; batchIndex < snapshots.Count; batchIndex++)
		{
			int legalCount = Math.Min(legalActionsBatch[batchIndex].Count, CombatModelMetadata.MaxActions);
			float[] logits = new float[legalCount];
			for (int actionIndex = 0; actionIndex < legalCount; actionIndex++)
			{
				logits[actionIndex] = logitsTensor[batchIndex, actionIndex];
			}

			float value = 0f;
			if (useContinuationValue && continuationTensor != null)
			{
				value = ReadScalar(continuationTensor, batchIndex) * 2f - 1f;
			}
			else if (valueTensor != null)
			{
				value = ReadScalar(valueTensor, batchIndex);
			}

			output[batchIndex] = new CombatEvaluationResult
			{
				PolicyLogits = logits,
				Value = value
			};
		}

		MaybeWriteDebugDump(snapshots[0], legalActionsBatch[0], encoded[0], output[0]);

		return output;
	}

	private void MaybeWriteDebugDump(
		FullRunSimulationStateSnapshot snapshot,
		IReadOnlyList<FullRunSimulationLegalAction> legalActions,
		CombatEncodedFeatures encoded,
		CombatEvaluationResult result)
	{
		string? dumpPath = Environment.GetEnvironmentVariable("STS2AI_DEBUG_COMBAT_FEATURES_PATH");
		if (string.IsNullOrWhiteSpace(dumpPath))
		{
			return;
		}

		if (Interlocked.Exchange(ref _debugDumpWritten, 1) != 0)
		{
			return;
		}

		try
		{
			string fullPath = Path.GetFullPath(dumpPath);
			string? directory = Path.GetDirectoryName(fullPath);
			if (!string.IsNullOrWhiteSpace(directory))
			{
				Directory.CreateDirectory(directory);
			}

			var payload = new
			{
				state_type = snapshot.StateType,
				total_floor = snapshot.TotalFloor,
				metadata = new
				{
					Metadata.HasDeckInputs,
					Metadata.HasPileInputs,
					Metadata.HasExtraScalarsInput,
					Metadata.HasValueOutput,
					Metadata.HasContinuationOutput,
					Metadata.PolicyOutputName,
					Metadata.ValueOutputName,
					Metadata.ContinuationOutputName,
				},
				legal_actions = legalActions.Select(static action => new
				{
					action = action.Action,
					label = action.Label,
					card_index = action.CardIndex,
					index = action.Index,
					target_id = action.TargetId,
					card_id = action.CardId,
				}).ToArray(),
				encoded = new
				{
					scalars = encoded.Scalars,
					extra_scalars = encoded.ExtraScalars,
					hand_ids = encoded.HandIds,
					hand_aux = encoded.HandAux,
					hand_mask = encoded.HandMask,
					enemy_ids = encoded.EnemyIds,
					enemy_aux = encoded.EnemyAux,
					enemy_mask = encoded.EnemyMask,
					action_type_ids = encoded.ActionTypeIds,
					target_card_ids = encoded.TargetCardIds,
					target_enemy_ids = encoded.TargetEnemyIds,
					action_mask = encoded.ActionMask,
					deck_ids = encoded.DeckIds,
					deck_aux = encoded.DeckAux,
					deck_mask = encoded.DeckMask,
					draw_pile_ids = encoded.DrawPileIds,
					draw_pile_aux = encoded.DrawPileAux,
					draw_pile_mask = encoded.DrawPileMask,
					discard_pile_ids = encoded.DiscardPileIds,
					discard_pile_aux = encoded.DiscardPileAux,
					discard_pile_mask = encoded.DiscardPileMask,
					exhaust_pile_ids = encoded.ExhaustPileIds,
					exhaust_pile_aux = encoded.ExhaustPileAux,
					exhaust_pile_mask = encoded.ExhaustPileMask,
				},
				output = new
				{
					policy_logits = result.PolicyLogits,
					value = result.Value,
				},
			};

			File.WriteAllText(
				fullPath,
				JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
		}
		catch (Exception ex)
		{
			Console.Error.WriteLine($"[ORT] Failed to write debug combat feature dump: {ex.Message}");
		}
	}

	private List<NamedOnnxValue> BuildInputs(IReadOnlyList<CombatEncodedFeatures> batch)
	{
		int batchSize = batch.Count;
		List<NamedOnnxValue> inputs = new()
		{
			NamedOnnxValue.CreateFromTensor("scalars",
				new DenseTensor<float>(Stack(batch, static encoded => encoded.Scalars), new[] { batchSize, CombatModelMetadata.CombatScalarDim })),
			NamedOnnxValue.CreateFromTensor("hand_ids",
				new DenseTensor<long>(Stack(batch, static encoded => encoded.HandIds), new[] { batchSize, CombatModelMetadata.MaxHandSize })),
			NamedOnnxValue.CreateFromTensor("hand_aux",
				new DenseTensor<float>(Stack(batch, static encoded => encoded.HandAux), new[] { batchSize, CombatModelMetadata.MaxHandSize, CombatModelMetadata.CardAuxDim })),
			NamedOnnxValue.CreateFromTensor("hand_mask",
				new DenseTensor<float>(Stack(batch, static encoded => encoded.HandMask), new[] { batchSize, CombatModelMetadata.MaxHandSize })),
			NamedOnnxValue.CreateFromTensor("enemy_ids",
				new DenseTensor<long>(Stack(batch, static encoded => encoded.EnemyIds), new[] { batchSize, CombatModelMetadata.MaxEnemies })),
			NamedOnnxValue.CreateFromTensor("enemy_aux",
				new DenseTensor<float>(Stack(batch, static encoded => encoded.EnemyAux), new[] { batchSize, CombatModelMetadata.MaxEnemies, CombatModelMetadata.EnemyAuxDim })),
			NamedOnnxValue.CreateFromTensor("enemy_mask",
				new DenseTensor<float>(Stack(batch, static encoded => encoded.EnemyMask), new[] { batchSize, CombatModelMetadata.MaxEnemies })),
			NamedOnnxValue.CreateFromTensor("action_type_ids",
				new DenseTensor<long>(Stack(batch, static encoded => encoded.ActionTypeIds), new[] { batchSize, CombatModelMetadata.MaxActions })),
			NamedOnnxValue.CreateFromTensor("target_card_ids",
				new DenseTensor<long>(Stack(batch, static encoded => encoded.TargetCardIds), new[] { batchSize, CombatModelMetadata.MaxActions })),
			NamedOnnxValue.CreateFromTensor("target_enemy_ids",
				new DenseTensor<long>(Stack(batch, static encoded => encoded.TargetEnemyIds), new[] { batchSize, CombatModelMetadata.MaxActions })),
			NamedOnnxValue.CreateFromTensor("action_mask",
				new DenseTensor<float>(Stack(batch, static encoded => encoded.ActionMask), new[] { batchSize, CombatModelMetadata.MaxActions })),
		};

		if (Metadata.HasExtraScalarsInput)
		{
			inputs.Add(NamedOnnxValue.CreateFromTensor("extra_scalars",
				new DenseTensor<float>(Stack(batch, static encoded => encoded.ExtraScalars), new[] { batchSize, CombatModelMetadata.CombatExtraScalarDim })));
		}

		if (Metadata.HasDeckInputs)
		{
			inputs.Add(NamedOnnxValue.CreateFromTensor("deck_ids",
				new DenseTensor<long>(Stack(batch, static encoded => encoded.DeckIds ?? new long[CombatModelMetadata.MaxDeckSize]), new[] { batchSize, CombatModelMetadata.MaxDeckSize })));
			inputs.Add(NamedOnnxValue.CreateFromTensor("deck_aux",
				new DenseTensor<float>(Stack(batch, static encoded => encoded.DeckAux ?? new float[CombatModelMetadata.MaxDeckSize * CombatModelMetadata.CardAuxDim]), new[] { batchSize, CombatModelMetadata.MaxDeckSize, CombatModelMetadata.CardAuxDim })));
			inputs.Add(NamedOnnxValue.CreateFromTensor("deck_mask",
				new DenseTensor<float>(Stack(batch, static encoded => encoded.DeckMask ?? new float[CombatModelMetadata.MaxDeckSize]), new[] { batchSize, CombatModelMetadata.MaxDeckSize })));
		}

		if (Metadata.HasPileInputs)
		{
			inputs.Add(NamedOnnxValue.CreateFromTensor("draw_pile_ids",
				new DenseTensor<long>(Stack(batch, static encoded => encoded.DrawPileIds ?? new long[CombatModelMetadata.MaxPileSize]), new[] { batchSize, CombatModelMetadata.MaxPileSize })));
			inputs.Add(NamedOnnxValue.CreateFromTensor("draw_pile_aux",
				new DenseTensor<float>(Stack(batch, static encoded => encoded.DrawPileAux ?? new float[CombatModelMetadata.MaxPileSize * CombatModelMetadata.CardAuxDim]), new[] { batchSize, CombatModelMetadata.MaxPileSize, CombatModelMetadata.CardAuxDim })));
			inputs.Add(NamedOnnxValue.CreateFromTensor("draw_pile_mask",
				new DenseTensor<float>(Stack(batch, static encoded => encoded.DrawPileMask ?? new float[CombatModelMetadata.MaxPileSize]), new[] { batchSize, CombatModelMetadata.MaxPileSize })));
			inputs.Add(NamedOnnxValue.CreateFromTensor("discard_pile_ids",
				new DenseTensor<long>(Stack(batch, static encoded => encoded.DiscardPileIds ?? new long[CombatModelMetadata.MaxPileSize]), new[] { batchSize, CombatModelMetadata.MaxPileSize })));
			inputs.Add(NamedOnnxValue.CreateFromTensor("discard_pile_aux",
				new DenseTensor<float>(Stack(batch, static encoded => encoded.DiscardPileAux ?? new float[CombatModelMetadata.MaxPileSize * CombatModelMetadata.CardAuxDim]), new[] { batchSize, CombatModelMetadata.MaxPileSize, CombatModelMetadata.CardAuxDim })));
			inputs.Add(NamedOnnxValue.CreateFromTensor("discard_pile_mask",
				new DenseTensor<float>(Stack(batch, static encoded => encoded.DiscardPileMask ?? new float[CombatModelMetadata.MaxPileSize]), new[] { batchSize, CombatModelMetadata.MaxPileSize })));
			inputs.Add(NamedOnnxValue.CreateFromTensor("exhaust_pile_ids",
				new DenseTensor<long>(Stack(batch, static encoded => encoded.ExhaustPileIds ?? new long[CombatModelMetadata.MaxPileSize]), new[] { batchSize, CombatModelMetadata.MaxPileSize })));
			inputs.Add(NamedOnnxValue.CreateFromTensor("exhaust_pile_aux",
				new DenseTensor<float>(Stack(batch, static encoded => encoded.ExhaustPileAux ?? new float[CombatModelMetadata.MaxPileSize * CombatModelMetadata.CardAuxDim]), new[] { batchSize, CombatModelMetadata.MaxPileSize, CombatModelMetadata.CardAuxDim })));
			inputs.Add(NamedOnnxValue.CreateFromTensor("exhaust_pile_mask",
				new DenseTensor<float>(Stack(batch, static encoded => encoded.ExhaustPileMask ?? new float[CombatModelMetadata.MaxPileSize]), new[] { batchSize, CombatModelMetadata.MaxPileSize })));
		}

		return inputs;
	}

	private static T[] Stack<T>(IReadOnlyList<CombatEncodedFeatures> batch, Func<CombatEncodedFeatures, T[]> selector)
	{
		int width = selector(batch[0]).Length;
		T[] stacked = new T[batch.Count * width];
		for (int i = 0; i < batch.Count; i++)
		{
			T[] row = selector(batch[i]);
			if (row.Length != width)
			{
				throw new InvalidOperationException("Inconsistent feature width within ORT batch.");
			}

			Array.Copy(row, 0, stacked, i * width, width);
		}

		return stacked;
	}

	private static float ReadScalar(Tensor<float> tensor, int batchIndex)
	{
		return tensor.Rank switch
		{
			0 => tensor.GetValue(0),
			1 => tensor[batchIndex],
			_ => tensor[batchIndex, 0],
		};
	}

	private static SessionBootstrapResult CreateSession(string onnxPath)
	{
		string requestedDevice = (Environment.GetEnvironmentVariable("STS2AI_ORT_DEVICE") ?? "auto").Trim().ToLowerInvariant();
		int cudaDeviceId = ParseNonNegativeInt(Environment.GetEnvironmentVariable("STS2AI_ORT_CUDA_DEVICE_ID"), fallback: 0);
		OrtDevicePreference preference = ParseDevicePreference(requestedDevice);

		if (preference == OrtDevicePreference.Cpu)
		{
			return CreateCpuSession(onnxPath, requestedDevice, fellBackToCpu: false, fallbackReason: null);
		}

		try
		{
			return CreateCudaSession(onnxPath, requestedDevice, cudaDeviceId);
		}
		catch (Exception ex) when (preference == OrtDevicePreference.Auto)
		{
			return CreateCpuSession(onnxPath, requestedDevice, fellBackToCpu: true, fallbackReason: ex.Message);
		}
	}

	private static SessionBootstrapResult CreateCpuSession(
		string onnxPath,
		string requestedDevice,
		bool fellBackToCpu,
		string? fallbackReason)
	{
		SessionOptions options = CreateCpuSessionOptions();
		return new SessionBootstrapResult
		{
			Session = new InferenceSession(onnxPath, options),
			RequestedDevice = requestedDevice,
			ExecutionProviderName = "CPUExecutionProvider",
			FellBackToCpu = fellBackToCpu,
			FallbackReason = fallbackReason,
		};
	}

	private static SessionBootstrapResult CreateCudaSession(string onnxPath, string requestedDevice, int cudaDeviceId)
	{
		SessionOptions options = CreateCudaSessionOptions(cudaDeviceId);
		return new SessionBootstrapResult
		{
			Session = new InferenceSession(onnxPath, options),
			RequestedDevice = requestedDevice,
			ExecutionProviderName = "CUDAExecutionProvider",
			FellBackToCpu = false,
			FallbackReason = null,
		};
	}

	private static SessionOptions CreateCpuSessionOptions()
	{
		SessionOptions options = new SessionOptions();
		ApplyCommonSessionOptions(options);
		return options;
	}

	private static SessionOptions CreateCudaSessionOptions(int cudaDeviceId)
	{
		PrepareCudaDependencySearchPath();
		SessionOptions options = SessionOptions.MakeSessionOptionWithCudaProvider(cudaDeviceId);
		ApplyCommonSessionOptions(options);
		return options;
	}

	private static void ApplyCommonSessionOptions(SessionOptions options)
	{
		options.IntraOpNumThreads = 1;
		options.InterOpNumThreads = 1;
		options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
		options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
	}

	private static OrtDevicePreference ParseDevicePreference(string requestedDevice)
	{
		return requestedDevice switch
		{
			"cpu" => OrtDevicePreference.Cpu,
			"cuda" or "gpu" => OrtDevicePreference.Cuda,
			"" or "auto" => OrtDevicePreference.Auto,
			_ => OrtDevicePreference.Auto,
		};
	}

	private static int ParseNonNegativeInt(string? rawValue, int fallback)
	{
		if (int.TryParse(rawValue, out int parsed) && parsed >= 0)
		{
			return parsed;
		}

		return fallback;
	}

	private static void PrepareCudaDependencySearchPath()
	{
		lock (_cudaDependencyPathLock)
		{
			if (_cudaDependencyPathPrepared)
			{
				return;
			}

			_cudaDependencyPathPrepared = true;
			foreach (string directory in EnumerateCudaDependencyDirectories())
			{
				if (PrependDirectoryToPath(directory))
				{
					Console.Error.WriteLine($"[ORT] Added CUDA DLL search path: {directory}");
				}
			}
		}
	}

	private static IEnumerable<string> EnumerateCudaDependencyDirectories()
	{
		HashSet<string> seen = new(StringComparer.OrdinalIgnoreCase);

		foreach (string raw in SplitPathList(Environment.GetEnvironmentVariable("STS2AI_ORT_DLL_DIRS")))
		{
			string fullPath = NormalizeExistingDirectory(raw);
			if (fullPath.Length > 0 && seen.Add(fullPath))
			{
				yield return fullPath;
			}
		}

		string configuredTorchLib = NormalizeExistingDirectory(Environment.GetEnvironmentVariable("STS2AI_TORCH_LIB_DIR"));
		if (configuredTorchLib.Length > 0 && seen.Add(configuredTorchLib))
		{
			yield return configuredTorchLib;
		}

		string discoveredTorchLib = NormalizeExistingDirectory(TryResolveTorchLibDirectory());
		if (discoveredTorchLib.Length > 0 && seen.Add(discoveredTorchLib))
		{
			yield return discoveredTorchLib;
		}
	}

	private static IEnumerable<string> SplitPathList(string? rawValue)
	{
		if (string.IsNullOrWhiteSpace(rawValue))
		{
			yield break;
		}

		foreach (string part in rawValue.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
		{
			if (!string.IsNullOrWhiteSpace(part))
			{
				yield return part;
			}
		}
	}

	private static string NormalizeExistingDirectory(string? rawValue)
	{
		if (string.IsNullOrWhiteSpace(rawValue))
		{
			return string.Empty;
		}

		try
		{
			string fullPath = Path.GetFullPath(rawValue.Trim());
			return Directory.Exists(fullPath) ? fullPath : string.Empty;
		}
		catch
		{
			return string.Empty;
		}
	}

	private static bool PrependDirectoryToPath(string directory)
	{
		string currentPath = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
		string[] entries = currentPath.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
		if (entries.Any(entry => string.Equals(entry, directory, StringComparison.OrdinalIgnoreCase)))
		{
			return false;
		}

		string updatedPath = string.IsNullOrWhiteSpace(currentPath)
			? directory
			: directory + Path.PathSeparator + currentPath;
		Environment.SetEnvironmentVariable("PATH", updatedPath);
		return true;
	}

	private static string? TryResolveTorchLibDirectory()
	{
		foreach ((string fileName, string[] prefixArgs) candidate in EnumeratePythonCandidates())
		{
			string? resolved = TryQueryTorchLibDirectory(candidate.fileName, candidate.prefixArgs);
			if (!string.IsNullOrWhiteSpace(resolved))
			{
				return resolved;
			}
		}

		return null;
	}

	private static IEnumerable<(string fileName, string[] prefixArgs)> EnumeratePythonCandidates()
	{
		string? configuredPython = Environment.GetEnvironmentVariable("STS2AI_PYTHON_EXE");
		if (!string.IsNullOrWhiteSpace(configuredPython))
		{
			yield return (configuredPython.Trim(), Array.Empty<string>());
		}

		yield return ("python", Array.Empty<string>());
		yield return ("py", new[] { "-3" });
	}

	private static string? TryQueryTorchLibDirectory(string fileName, IReadOnlyList<string> prefixArgs)
	{
		const string ProbeCode =
			"from pathlib import Path; import torch; print((Path(torch.__file__).resolve().parent / 'lib'))";

		try
		{
			ProcessStartInfo startInfo = new ProcessStartInfo
			{
				FileName = fileName,
				RedirectStandardOutput = true,
				RedirectStandardError = true,
				UseShellExecute = false,
				CreateNoWindow = true,
			};

			foreach (string arg in prefixArgs)
			{
				startInfo.ArgumentList.Add(arg);
			}

			startInfo.ArgumentList.Add("-c");
			startInfo.ArgumentList.Add(ProbeCode);

			using Process process = Process.Start(startInfo)!;
			if (!process.WaitForExit(5000))
			{
				try
				{
					process.Kill(entireProcessTree: true);
				}
				catch
				{
				}
				return null;
			}

			if (process.ExitCode != 0)
			{
				return null;
			}

			string output = process.StandardOutput.ReadToEnd().Trim();
			return string.IsNullOrWhiteSpace(output) ? null : output;
		}
		catch
		{
			return null;
		}
	}

	private static CombatModelMetadata InspectModel(InferenceSession session)
	{
		string policyOutputName = session.OutputMetadata.ContainsKey("policy_logits")
			? "policy_logits"
			: session.OutputMetadata.Keys.First();

		return new CombatModelMetadata
		{
			HasDeckInputs = session.InputMetadata.ContainsKey("deck_ids"),
			HasPileInputs = session.InputMetadata.ContainsKey("draw_pile_ids"),
			HasExtraScalarsInput = session.InputMetadata.ContainsKey("extra_scalars"),
			HasValueOutput = session.OutputMetadata.ContainsKey("value"),
			HasContinuationOutput = session.OutputMetadata.ContainsKey("continuation"),
			PolicyOutputName = policyOutputName,
			ValueOutputName = session.OutputMetadata.ContainsKey("value") ? "value" : null,
			ContinuationOutputName = session.OutputMetadata.ContainsKey("continuation") ? "continuation" : null,
		};
	}

	public void Dispose()
	{
		_session.Dispose();
	}
}

internal sealed class OrtActorPolicy : IDisposable
{
	private readonly OrtCombatEvaluator _evaluator;
	private readonly bool _argmax;

	public OrtActorPolicy(string onnxPath, bool argmax = true, string? vocabPath = null)
	{
		_evaluator = new OrtCombatEvaluator(onnxPath, vocabPath);
		_argmax = argmax;
	}

	public CombatModelMetadata Metadata => _evaluator.Metadata;

	public string ExecutionProviderName => _evaluator.ExecutionProviderName;

	public string RequestedDevice => _evaluator.RequestedDevice;

	public bool FellBackToCpu => _evaluator.FellBackToCpu;

	public CombatEvaluationResult Evaluate(
		FullRunSimulationStateSnapshot snapshot,
		IReadOnlyList<FullRunSimulationLegalAction> legalActions,
		bool useContinuationValue)
	{
		return _evaluator.Evaluate(snapshot, legalActions, useContinuationValue);
	}

	public IReadOnlyList<CombatEvaluationResult> EvaluateBatch(
		IReadOnlyList<FullRunSimulationStateSnapshot> snapshots,
		IReadOnlyList<IReadOnlyList<FullRunSimulationLegalAction>> legalActionsBatch,
		bool useContinuationValue)
	{
		return _evaluator.EvaluateBatch(snapshots, legalActionsBatch, useContinuationValue);
	}

	public (int ActionIndex, float[] Logits) SelectAction(
		FullRunSimulationStateSnapshot snapshot,
		BinarySessionState? session,
		Random rng)
	{
		_ = session;
		CombatEvaluationResult result = _evaluator.Evaluate(snapshot, snapshot.LegalActions, useContinuationValue: false);
		int legalCount = Math.Min(snapshot.LegalActions.Count, result.PolicyLogits.Length);
		int actionIndex = _argmax || rng == null
			? Argmax(result.PolicyLogits, legalCount)
			: Sample(result.PolicyLogits, legalCount, rng);
		return (actionIndex, result.PolicyLogits);
	}

	public void Dispose()
	{
		_evaluator.Dispose();
	}

	private static int Argmax(IReadOnlyList<float> logits, int count)
	{
		if (count <= 0)
		{
			return 0;
		}

		int best = 0;
		float bestValue = float.NegativeInfinity;
		for (int i = 0; i < count; i++)
		{
			if (logits[i] > bestValue)
			{
				bestValue = logits[i];
				best = i;
			}
		}

		return best;
	}

	private static int Sample(IReadOnlyList<float> logits, int count, Random rng)
	{
		if (count <= 0)
		{
			return 0;
		}

		float maxLogit = float.NegativeInfinity;
		for (int i = 0; i < count; i++)
		{
			maxLogit = Math.Max(maxLogit, logits[i]);
		}

		float[] probs = new float[count];
		float total = 0f;
		for (int i = 0; i < count; i++)
		{
			probs[i] = MathF.Exp(logits[i] - maxLogit);
			total += probs[i];
		}

		if (!(total > 0f))
		{
			return 0;
		}

		float threshold = (float)rng.NextDouble() * total;
		float cumulative = 0f;
		for (int i = 0; i < count; i++)
		{
			cumulative += probs[i];
			if (threshold <= cumulative)
			{
				return i;
			}
		}

		return count - 1;
	}
}
