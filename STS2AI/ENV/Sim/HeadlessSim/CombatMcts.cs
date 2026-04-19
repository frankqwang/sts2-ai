using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Simulation;
using MegaCrit.Sts2.Core.Training;

namespace HeadlessSim;

internal sealed class CombatMctsConfig
{
	public int NumSimulations { get; init; } = 50;

	public float CPuct { get; init; } = 1.5f;

	public float DirichletAlpha { get; init; } = 0.3f;

	public float DirichletFraction { get; init; } = 0.25f;

	public int MaxStepBudget { get; init; } = 200;

	public string FinalActionMode { get; init; } = "visit";

	public int FinalActionTopK { get; init; } = 3;

	public float FinalActionQWeight { get; init; } = 0.35f;

	public bool UseContinuationValue { get; init; }

	public bool EnableDebugTrace { get; init; }
}

internal sealed class CombatMctsResult
{
	public int ActionIndex { get; init; }

	public int[] VisitCounts { get; init; } = Array.Empty<int>();

	public float[] VisitProbs { get; init; } = Array.Empty<float>();

	public float[] QValues { get; init; } = Array.Empty<float>();

	public float[] Priors { get; init; } = Array.Empty<float>();

	public float RootValue { get; init; }

	public float SearchMs { get; init; }

	public bool RestoredOk { get; init; }

	public int SnapshotCount { get; init; }

	public CombatMctsBreakdown Breakdown { get; init; } = new();

	public string? DebugTraceJson { get; init; }
}

internal sealed class CombatMctsBreakdown
{
	public int SimulationCount { get; set; }

	public int SaveStateCount { get; set; }

	public int LoadStateCount { get; set; }

	public int DeleteStateCount { get; set; }

	public int StepCount { get; set; }

	public int AdvanceStateCount { get; set; }

	public int EvalCallCount { get; set; }

	public int EvalBatchCount { get; set; }

	public int EvalStateCount { get; set; }

	public int SelectChildCount { get; set; }

	public int BackpropCount { get; set; }

	public float SaveStateMs { get; set; }

	public float LoadStateMs { get; set; }

	public float DeleteStateMs { get; set; }

	public float StepMs { get; set; }

	public float AdvanceStateMs { get; set; }

	public float EvalMs { get; set; }

	public float SelectionMs { get; set; }

	public float BackpropMs { get; set; }
}

internal sealed class CombatMctsNode
{
	public CombatMctsNode? Parent { get; init; }

	public FullRunSimulationActionRequest? Action { get; init; }

	public float Prior { get; set; }

	public List<CombatMctsNode> Children { get; } = new();

	public int VisitCount { get; set; }

	public float TotalValue { get; set; }

	public bool IsExpanded { get; set; }

	public bool IsTerminal { get; set; }

	public float TerminalValue { get; set; }

	public string? StateId { get; set; }

	public int StepCount { get; set; }

	public float QValue => VisitCount == 0 ? 0f : TotalValue / VisitCount;

	public void Expand(IReadOnlyList<FullRunSimulationLegalAction> legalActions, IReadOnlyList<float> priors)
	{
		Children.Clear();
		for (int i = 0; i < legalActions.Count; i++)
		{
			Children.Add(new CombatMctsNode
			{
				Parent = this,
				Action = BuildActionRequest(legalActions[i]),
				Prior = i < priors.Count ? priors[i] : 0f,
				StepCount = StepCount + 1,
			});
		}

		IsExpanded = true;
	}

	private static FullRunSimulationActionRequest BuildActionRequest(FullRunSimulationLegalAction legalAction)
	{
		return new FullRunSimulationActionRequest
		{
			Action = legalAction.Action ?? string.Empty,
			Type = legalAction.Action ?? string.Empty,
			Index = legalAction.Index,
			Col = legalAction.Col,
			Row = legalAction.Row,
			Value = legalAction.Label,
			CardIndex = legalAction.CardIndex,
			HandIndex = legalAction.CardIndex,
			Slot = legalAction.Slot,
			TargetId = legalAction.TargetId,
			Target = legalAction.Target,
		};
	}
}

internal sealed class CombatMctsSimulationTrace
{
	public int Simulation { get; init; }

	public List<string> Path { get; init; } = new();

	public string? LeafStateType { get; set; }

	public string? LeafSignature { get; set; }

	public string? LeafSummary { get; set; }

	public int LeafLegalActionCount { get; set; }

	public string? Outcome { get; set; }

	public float LeafValue { get; set; }
}

internal sealed class CombatMctsSearchEngine
{
	private const int EvalBatchSize = 8;
	private static readonly HashSet<string> CombatActiveStates = new(StringComparer.OrdinalIgnoreCase)
	{
		"combat",
		"monster",
		"elite",
		"boss",
		"hand_select",
		"card_select",
		"combat_pending",
		"combat_start_pending",
	};

	private sealed class PendingLeaf
	{
		public CombatMctsNode Node { get; init; } = new CombatMctsNode();

		public FullRunSimulationStateSnapshot Snapshot { get; init; } = new FullRunSimulationStateSnapshot();

		public IReadOnlyList<FullRunSimulationLegalAction> LegalActions { get; init; } = Array.Empty<FullRunSimulationLegalAction>();

		public CombatMctsSimulationTrace? Trace { get; init; }
	}

	private readonly FullRunTrainingEnvService _service;
	private readonly OrtActorPolicy _evaluator;
	private readonly Random _rng;

	public CombatMctsSearchEngine(FullRunTrainingEnvService service, OrtActorPolicy evaluator, Random? rng = null)
	{
		_service = service;
		_evaluator = evaluator;
		_rng = rng ?? new Random(42);
	}

	public async Task<CombatMctsResult> SearchAsync(
		FullRunSimulationStateSnapshot rootSnapshot,
		CombatMctsConfig config)
	{
		Stopwatch stopwatch = Stopwatch.StartNew();
		CombatMctsBreakdown breakdown = new();
		CombatMctsNode root = new CombatMctsNode();
		List<string> createdStateIds = new();
		string? rootStateId = null;
		string rootSignature = string.Empty;
		bool restoredOk = false;
		List<CombatMctsSimulationTrace>? debugTrace = config.EnableDebugTrace ? new List<CombatMctsSimulationTrace>() : null;

		try
		{
			rootSnapshot = await AdvanceCombatStateAsync(rootSnapshot, breakdown);
			rootSignature = BuildStateSignature(rootSnapshot);
			rootStateId = SaveState(breakdown, includeFullFallback: true);
			createdStateIds.Add(rootStateId);
			root.StateId = rootStateId;

			if (IsTerminal(rootSnapshot, out float terminalValue))
			{
				root.IsTerminal = true;
				root.TerminalValue = terminalValue;
				FullRunSimulationStateSnapshot restoredTerminal = await LoadState(rootStateId, breakdown);
				restoredOk = BuildStateSignature(restoredTerminal) == rootSignature;
				return new CombatMctsResult
				{
					ActionIndex = 0,
					RootValue = terminalValue,
					SearchMs = (float)stopwatch.Elapsed.TotalMilliseconds,
					RestoredOk = restoredOk,
					SnapshotCount = createdStateIds.Count,
					Breakdown = breakdown,
				};
			}

			IReadOnlyList<FullRunSimulationLegalAction> rootLegalActions = rootSnapshot.LegalActions;
			CombatEvaluationResult rootEval = Evaluate(rootSnapshot, rootLegalActions, config.UseContinuationValue, breakdown);
			float[] rootPriors = Softmax(rootEval.PolicyLogits, rootLegalActions.Count);
			if (config.DirichletAlpha > 0f && config.DirichletFraction > 0f && rootPriors.Length > 0)
			{
				float[] noise = SampleDirichlet(rootPriors.Length, config.DirichletAlpha);
				for (int i = 0; i < rootPriors.Length; i++)
				{
					rootPriors[i] = (1f - config.DirichletFraction) * rootPriors[i] + config.DirichletFraction * noise[i];
				}
				NormalizeInPlace(rootPriors);
			}
			root.Expand(rootLegalActions, rootPriors);

			List<PendingLeaf> pendingLeaves = new();
			List<(CombatMctsNode Node, float Value)> pendingBackprop = new();
			int simulations = 0;

			while (simulations < Math.Max(1, config.NumSimulations))
			{
				breakdown.SimulationCount += 1;
				await LoadState(rootStateId, breakdown);
				FullRunSimulationStateSnapshot currentSnapshot = rootSnapshot;
				CombatMctsNode node = root;
				CombatMctsSimulationTrace? simTrace = debugTrace != null
					? new CombatMctsSimulationTrace { Simulation = simulations + 1 }
					: null;

				while (node.IsExpanded && !node.IsTerminal && node.Children.Count > 0)
				{
					double selectionStart = GetTimestampMs();
					CombatMctsNode child = SelectChild(node, config.CPuct);
					breakdown.SelectionMs += (float)(GetTimestampMs() - selectionStart);
					breakdown.SelectChildCount += 1;
					simTrace?.Path.Add(BuildActionSignature(child.Action, child.StateId != null ? "load" : "step"));
					if (child.StateId != null)
					{
						currentSnapshot = await LoadState(child.StateId, breakdown);
						node = child;
						continue;
					}

					if (child.StepCount > Math.Max(1, config.MaxStepBudget))
					{
						child.IsTerminal = true;
						child.TerminalValue = -1f;
						node = child;
						break;
					}

					FullRunSimulationStepResult stepResult = await StepAsync(child.Action!, breakdown);
					currentSnapshot = stepResult.State ?? _service.GetState();
					currentSnapshot = await AdvanceCombatStateAsync(currentSnapshot, breakdown);
					node = child;
				}

				if (node.IsTerminal)
				{
					RecordLeafTrace(simTrace, currentSnapshot, "node_terminal", node.TerminalValue);
					pendingBackprop.Add((node, node.TerminalValue));
				}
				else if (IsTerminal(currentSnapshot, out float leafTerminalValue))
				{
					node.IsTerminal = true;
					node.TerminalValue = leafTerminalValue;
					RecordLeafTrace(simTrace, currentSnapshot, "snapshot_terminal", leafTerminalValue);
					pendingBackprop.Add((node, leafTerminalValue));
				}
				else if (currentSnapshot.LegalActions.Count == 0)
				{
					node.IsTerminal = true;
					node.TerminalValue = -1f;
					RecordLeafTrace(simTrace, currentSnapshot, "no_legal_actions", -1f);
					pendingBackprop.Add((node, -1f));
				}
				else
				{
					if (node.StateId == null)
					{
						node.StateId = SaveState(breakdown);
						createdStateIds.Add(node.StateId);
					}

					node.VisitCount += 1;
					node.TotalValue -= 1f;
					pendingLeaves.Add(new PendingLeaf
					{
						Node = node,
						Snapshot = currentSnapshot,
						LegalActions = currentSnapshot.LegalActions,
						Trace = simTrace,
					});
				}

				simulations++;
				if (pendingLeaves.Count >= EvalBatchSize || (simulations >= config.NumSimulations && pendingLeaves.Count > 0))
				{
					IReadOnlyList<CombatEvaluationResult> batchEval = EvaluateBatch(
						pendingLeaves.Select(static leaf => leaf.Snapshot).ToArray(),
						pendingLeaves.Select(static leaf => leaf.LegalActions).ToArray(),
						config.UseContinuationValue,
						breakdown);

					for (int i = 0; i < pendingLeaves.Count; i++)
					{
						PendingLeaf leaf = pendingLeaves[i];
						CombatEvaluationResult evaluation = batchEval[i];
						leaf.Node.VisitCount -= 1;
						leaf.Node.TotalValue += 1f;
						leaf.Node.Expand(leaf.LegalActions, Softmax(evaluation.PolicyLogits, leaf.LegalActions.Count));
						RecordLeafTrace(leaf.Trace, leaf.Snapshot, "expanded_leaf", evaluation.Value);
						pendingBackprop.Add((leaf.Node, evaluation.Value));
					}

					pendingLeaves.Clear();
				}

				foreach ((CombatMctsNode backpropNode, float value) in pendingBackprop)
				{
					double backpropStart = GetTimestampMs();
					Backpropagate(backpropNode, value);
					breakdown.BackpropMs += (float)(GetTimestampMs() - backpropStart);
					breakdown.BackpropCount += 1;
				}
				pendingBackprop.Clear();
				if (simTrace != null)
				{
					debugTrace!.Add(simTrace);
				}
			}

			FullRunSimulationStateSnapshot restored = await LoadState(rootStateId, breakdown);
			restoredOk = BuildStateSignature(restored) == rootSignature;
			return BuildResult(root, rootLegalActions.Count, config, (float)stopwatch.Elapsed.TotalMilliseconds, restoredOk, createdStateIds.Count, breakdown, debugTrace);
		}
		finally
		{
			foreach (string stateId in createdStateIds)
			{
				try
				{
					DeleteState(stateId, breakdown);
				}
				catch
				{
				}
			}
		}
	}

	private static CombatMctsResult BuildResult(
		CombatMctsNode root,
		int legalCount,
		CombatMctsConfig config,
		float searchMs,
		bool restoredOk,
		int snapshotCount,
		CombatMctsBreakdown breakdown,
		List<CombatMctsSimulationTrace>? debugTrace)
	{
		int count = Math.Min(legalCount, root.Children.Count);
		int[] visitCounts = new int[count];
		float[] visitProbs = new float[count];
		float[] qValues = new float[count];
		float[] priors = new float[count];
		int totalVisits = Math.Max(1, root.Children.Sum(static child => child.VisitCount));

		for (int i = 0; i < count; i++)
		{
			CombatMctsNode child = root.Children[i];
			visitCounts[i] = child.VisitCount;
			visitProbs[i] = child.VisitCount / (float)totalVisits;
			qValues[i] = child.QValue;
			priors[i] = child.Prior;
		}

		return new CombatMctsResult
		{
			ActionIndex = SelectFinalAction(root.Children, config),
			VisitCounts = visitCounts,
			VisitProbs = visitProbs,
			QValues = qValues,
			Priors = priors,
			RootValue = root.QValue,
			SearchMs = searchMs,
			RestoredOk = restoredOk,
			SnapshotCount = snapshotCount,
			Breakdown = breakdown,
			DebugTraceJson = debugTrace != null && debugTrace.Count > 0
				? JsonSerializer.Serialize(debugTrace)
				: null,
		};
	}

	private static void RecordLeafTrace(
		CombatMctsSimulationTrace? trace,
		FullRunSimulationStateSnapshot snapshot,
		string outcome,
		float value)
	{
		if (trace == null)
		{
			return;
		}

		trace.LeafStateType = snapshot.StateType;
		trace.LeafSignature = BuildStateSignature(snapshot);
		trace.LeafSummary = BuildDetailedStateSummary(snapshot);
		trace.LeafLegalActionCount = snapshot.LegalActions.Count;
		trace.Outcome = outcome;
		trace.LeafValue = value;
	}

	private static string BuildActionSignature(FullRunSimulationActionRequest? action, string source)
	{
		if (action == null)
		{
			return $"<null>:{source}";
		}

		return string.Join(":",
			action.Action ?? string.Empty,
			action.CardIndex?.ToString() ?? "-",
			action.TargetId?.ToString() ?? "-",
			action.Slot?.ToString() ?? "-",
			source);
	}

	private string SaveState(CombatMctsBreakdown breakdown, bool includeFullFallback = false)
	{
		double start = GetTimestampMs();
		string stateId = _service.SaveSearchState(includeFullFallback);
		breakdown.SaveStateMs += (float)(GetTimestampMs() - start);
		breakdown.SaveStateCount += 1;
		return stateId;
	}

	private async Task<FullRunSimulationStateSnapshot> LoadState(string stateId, CombatMctsBreakdown breakdown)
	{
		double start = GetTimestampMs();
		FullRunSimulationStateSnapshot snapshot = await _service.LoadState(stateId);
		breakdown.LoadStateMs += (float)(GetTimestampMs() - start);
		breakdown.LoadStateCount += 1;
		return snapshot;
	}

	private void DeleteState(string stateId, CombatMctsBreakdown breakdown)
	{
		double start = GetTimestampMs();
		_service.DeleteState(stateId);
		breakdown.DeleteStateMs += (float)(GetTimestampMs() - start);
		breakdown.DeleteStateCount += 1;
	}

	private async Task<FullRunSimulationStepResult> StepAsync(FullRunSimulationActionRequest action, CombatMctsBreakdown breakdown)
	{
		double start = GetTimestampMs();
		FullRunSimulationStepResult result = await _service.StepAsync(action);
		breakdown.StepMs += (float)(GetTimestampMs() - start);
		breakdown.StepCount += 1;
		return result;
	}

	private CombatEvaluationResult Evaluate(
		FullRunSimulationStateSnapshot snapshot,
		IReadOnlyList<FullRunSimulationLegalAction> legalActions,
		bool useContinuationValue,
		CombatMctsBreakdown breakdown)
	{
		double start = GetTimestampMs();
		CombatEvaluationResult result = _evaluator.Evaluate(snapshot, legalActions, useContinuationValue);
		breakdown.EvalMs += (float)(GetTimestampMs() - start);
		breakdown.EvalCallCount += 1;
		breakdown.EvalStateCount += 1;
		return result;
	}

	private IReadOnlyList<CombatEvaluationResult> EvaluateBatch(
		IReadOnlyList<FullRunSimulationStateSnapshot> snapshots,
		IReadOnlyList<IReadOnlyList<FullRunSimulationLegalAction>> legalActionsBatch,
		bool useContinuationValue,
		CombatMctsBreakdown breakdown)
	{
		double start = GetTimestampMs();
		IReadOnlyList<CombatEvaluationResult> result = _evaluator.EvaluateBatch(snapshots, legalActionsBatch, useContinuationValue);
		breakdown.EvalMs += (float)(GetTimestampMs() - start);
		breakdown.EvalBatchCount += 1;
		breakdown.EvalStateCount += snapshots.Count;
		return result;
	}

	private static double GetTimestampMs()
	{
		return Stopwatch.GetTimestamp() * 1000.0 / Stopwatch.Frequency;
	}

	private static int SelectFinalAction(IReadOnlyList<CombatMctsNode> children, CombatMctsConfig config)
	{
		if (children.Count == 0)
		{
			return 0;
		}

		if (string.Equals(config.FinalActionMode, "visit_q_blend", StringComparison.OrdinalIgnoreCase))
		{
			List<(CombatMctsNode Node, int Index)> ranked = children
				.Select((child, index) => (Node: child, Index: index))
				.OrderByDescending(static item => item.Node.VisitCount)
				.ThenByDescending(static item => item.Node.Prior)
				.ThenByDescending(static item => item.Node.QValue)
				.Take(Math.Max(1, Math.Min(config.FinalActionTopK, children.Count)))
				.ToList();

			float totalVisits = Math.Max(1, children.Sum(static child => child.VisitCount));
			float qWeight = Math.Clamp(config.FinalActionQWeight, 0f, 1f);
			return ranked
				.OrderByDescending(item =>
				{
					float visitFrac = item.Node.VisitCount / totalVisits;
					float qNorm = (item.Node.QValue + 1f) * 0.5f;
					return (1f - qWeight) * visitFrac + qWeight * qNorm;
				})
				.ThenByDescending(static item => item.Node.VisitCount)
				.ThenByDescending(static item => item.Node.QValue)
				.First().Index;
		}

		return children
			.Select((child, index) => (Node: child, Index: index))
			.OrderByDescending(static item => item.Node.VisitCount)
			.ThenByDescending(static item => item.Node.Prior)
			.First().Index;
	}

	private static void Backpropagate(CombatMctsNode leaf, float value)
	{
		CombatMctsNode? current = leaf;
		while (current != null)
		{
			current.VisitCount += 1;
			current.TotalValue += value;
			current = current.Parent;
		}
	}

	private static CombatMctsNode SelectChild(CombatMctsNode node, float cPuct)
	{
		CombatMctsNode? best = null;
		float bestScore = float.NegativeInfinity;
		float parentVisits = MathF.Sqrt(Math.Max(1, node.VisitCount));
		foreach (CombatMctsNode child in node.Children)
		{
			float exploration = cPuct * child.Prior * parentVisits / (1 + child.VisitCount);
			float score = child.QValue + exploration;
			if (score > bestScore)
			{
				bestScore = score;
				best = child;
			}
		}

		return best ?? node.Children[0];
	}

	private async Task<FullRunSimulationStateSnapshot> AdvanceCombatStateAsync(
		FullRunSimulationStateSnapshot snapshot,
		CombatMctsBreakdown breakdown)
	{
		double start = GetTimestampMs();
		breakdown.AdvanceStateCount += 1;
		for (int i = 0; i < 32; i++)
		{
			if (snapshot.IsTerminal || string.Equals(snapshot.StateType, "game_over", StringComparison.OrdinalIgnoreCase))
			{
				breakdown.AdvanceStateMs += (float)(GetTimestampMs() - start);
				return snapshot;
			}

			if (!CombatActiveStates.Contains(snapshot.StateType))
			{
				breakdown.AdvanceStateMs += (float)(GetTimestampMs() - start);
				return snapshot;
			}

			if (snapshot.LegalActions.Count > 0)
			{
				breakdown.AdvanceStateMs += (float)(GetTimestampMs() - start);
				return snapshot;
			}

			FullRunSimulationStepResult waitResult = await StepAsync(new FullRunSimulationActionRequest
			{
				Action = "wait",
				Type = "wait",
			}, breakdown);
			snapshot = waitResult.State ?? _service.GetState();
		}

		breakdown.AdvanceStateMs += (float)(GetTimestampMs() - start);
		return snapshot;
	}

	private static bool IsTerminal(FullRunSimulationStateSnapshot snapshot, out float value)
	{
		if (snapshot.IsTerminal || string.Equals(snapshot.StateType, "game_over", StringComparison.OrdinalIgnoreCase))
		{
			value = -1f;
			return true;
		}

		if (!CombatActiveStates.Contains(snapshot.StateType))
		{
			value = 1f;
			return true;
		}

		value = 0f;
		return false;
	}

	private static float[] Softmax(IReadOnlyList<float> logits, int count)
	{
		if (count <= 0)
		{
			return Array.Empty<float>();
		}

		float[] probs = new float[count];
		float maxLogit = float.NegativeInfinity;
		for (int i = 0; i < count; i++)
		{
			maxLogit = Math.Max(maxLogit, logits[i]);
		}

		float total = 0f;
		for (int i = 0; i < count; i++)
		{
			probs[i] = MathF.Exp(logits[i] - maxLogit);
			total += probs[i];
		}

		if (!(total > 0f))
		{
			float uniform = 1f / count;
			for (int i = 0; i < count; i++)
			{
				probs[i] = uniform;
			}
			return probs;
		}

		for (int i = 0; i < count; i++)
		{
			probs[i] /= total;
		}

		return probs;
	}

	private void NormalizeInPlace(float[] values)
	{
		float total = values.Sum();
		if (!(total > 0f))
		{
			float uniform = values.Length == 0 ? 0f : 1f / values.Length;
			for (int i = 0; i < values.Length; i++)
			{
				values[i] = uniform;
			}
			return;
		}

		for (int i = 0; i < values.Length; i++)
		{
			values[i] /= total;
		}
	}

	private float[] SampleDirichlet(int size, float alpha)
	{
		float[] samples = new float[size];
		float total = 0f;
		for (int i = 0; i < size; i++)
		{
			float gamma = SampleGamma(alpha);
			samples[i] = gamma;
			total += gamma;
		}

		if (!(total > 0f))
		{
			float uniform = 1f / size;
			for (int i = 0; i < size; i++)
			{
				samples[i] = uniform;
			}
			return samples;
		}

		for (int i = 0; i < size; i++)
		{
			samples[i] /= total;
		}

		return samples;
	}

	private float SampleGamma(float alpha)
	{
		if (alpha <= 0f)
		{
			return 0f;
		}

		if (alpha < 1f)
		{
			float u = (float)_rng.NextDouble();
			return SampleGamma(alpha + 1f) * MathF.Pow(u, 1f / alpha);
		}

		float d = alpha - 1f / 3f;
		float c = 1f / MathF.Sqrt(9f * d);
		while (true)
		{
			float x;
			float v;
			do
			{
				x = SampleStandardNormal();
				v = 1f + c * x;
			}
			while (v <= 0f);

			v = v * v * v;
			float u = (float)_rng.NextDouble();
			if (u < 1f - 0.0331f * x * x * x * x)
			{
				return d * v;
			}

			if (MathF.Log(u) < 0.5f * x * x + d * (1f - v + MathF.Log(v)))
			{
				return d * v;
			}
		}
	}

	private float SampleStandardNormal()
	{
		float u1 = 1f - (float)_rng.NextDouble();
		float u2 = 1f - (float)_rng.NextDouble();
		return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2);
	}

	private static string BuildStateSignature(FullRunSimulationStateSnapshot snapshot)
	{
		return string.Join("|",
			snapshot.StateType,
			snapshot.TotalFloor,
			snapshot.CurrentActIndex,
			snapshot.IsTerminal ? "1" : "0",
			snapshot.LegalActions.Count,
			string.Join(";", snapshot.LegalActions.Select(static action =>
				$"{action.Action}:{action.Index}:{action.CardIndex}:{action.TargetId}:{action.Slot}:{action.Label}")));
	}

	private static string BuildDetailedStateSummary(FullRunSimulationStateSnapshot snapshot)
	{
		CombatTrainingStateSnapshot? combat = snapshot.CachedCombatState;
		if (combat == null)
		{
			return $"state={snapshot.StateType}|floor={snapshot.TotalFloor}|legal={snapshot.LegalActions.Count}";
		}

		string playerSummary = combat.Player == null
			? "player=null"
			: $"player=hp:{combat.Player.CurrentHp}/{combat.Player.MaxHp},block:{combat.Player.Block},energy:{combat.Player.Energy}/{combat.Player.MaxEnergy}";
		string enemySummary = string.Join(";",
			combat.Enemies.Select(static enemy =>
				$"{enemy.CombatId ?? 0}:{enemy.Id ?? enemy.Name}:{enemy.CurrentHp}/{enemy.MaxHp}:b{enemy.Block}:alive={(enemy.IsAlive ? 1 : 0)}"));
		string handSummary = string.Join(";",
			combat.Hand.Select(static card =>
				$"{card.HandIndex}:{card.Id}:{card.EnergyCost}:{(card.CanPlay ? 1 : 0)}"));
		return string.Join("|",
			$"state={snapshot.StateType}",
			$"floor={snapshot.TotalFloor}",
			playerSummary,
			$"round={combat.RoundNumber}",
			$"side={combat.CurrentSide}",
			$"enemies={enemySummary}",
			$"hand={handSummary}",
			$"legal={snapshot.LegalActions.Count}");
	}
}
