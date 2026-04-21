using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Simulation;

namespace HeadlessSim;

internal static partial class Program
{
	// State types that require agent decision (return to Python)
	private static readonly HashSet<string> DecisionStateTypes = new(StringComparer.OrdinalIgnoreCase)
	{
		"map", "combat_rewards", "card_reward", "card_select", "relic_select",
		"shop", "rest_site", "campfire", "event", "treasure",
		"monster", "elite", "boss", "combat", "hand_select",
		"game_over", "menu",
	};

	private static bool IsDecisionState(FullRunSimulationStateSnapshot snapshot)
	{
		if (snapshot.IsTerminal || snapshot.StateType == "game_over")
		{
			return true;
		}

		if (snapshot.LegalActions.Count == 0)
		{
			return false;
		}

		return DecisionStateTypes.Contains(snapshot.StateType);
	}

	private static FullRunSimulationActionRequest BuildFullRunActionRequest(FullRunSimulationLegalAction action)
	{
		return new FullRunSimulationActionRequest
		{
			Action = action.Action ?? "",
			Index = action.Index,
			Col = action.Col,
			Row = action.Row,
			Slot = action.Slot,
			TargetId = action.TargetId,
			Target = action.Target,
			CardIndex = action.CardIndex,
			Value = action.Label,
		};
	}

	private static async Task<(FullRunSimulationStepResult Result, FullRunSimulationStateSnapshot Snapshot)> ExecuteFullRunStepAsync(
		FullRunTrainingEnvService service,
		RequestStateCache cache,
		FullRunSimulationActionRequest action,
		bool autoAdvanceToDecisionState)
	{
		FullRunSimulationStepResult result;
		using (FullRunSimulationDiagnostics.Measure("request.step.runtime_ms"))
		{
			result = await service.StepAsync(action);
		}

		FullRunSimulationStateSnapshot snapshot = result.State ?? GetSnapshot(service, cache);
		snapshot = autoAdvanceToDecisionState
			? await AutoAdvanceToDecisionStateAsync(service, cache, snapshot, maxAutoAdvance: 50)
			: await AutoAdvancePendingWaitStatesAsync(service, cache, snapshot, maxAutoAdvance: 30);
		return (result, snapshot);
	}

	private static async Task<FullRunSimulationStateSnapshot> AutoAdvanceToDecisionStateAsync(
		FullRunTrainingEnvService service,
		RequestStateCache cache,
		FullRunSimulationStateSnapshot snapshot,
		int maxAutoAdvance)
	{
		int autoAdvanceCount = 0;
		while (!IsDecisionState(snapshot) && autoAdvanceCount < maxAutoAdvance)
		{
			autoAdvanceCount++;
			FullRunSimulationDiagnostics.Increment("step.auto_advance");

			if (snapshot.LegalActions.Count == 0)
			{
				FullRunSimulationStepResult waitResult = await service.StepAsync(new FullRunSimulationActionRequest { Action = "wait" });
				snapshot = waitResult.State ?? GetSnapshot(service, cache);
				continue;
			}

			FullRunSimulationActionRequest autoAction = BuildFullRunActionRequest(snapshot.LegalActions[0]);
			FullRunSimulationStepResult autoResult = await service.StepAsync(autoAction);
			snapshot = autoResult.State ?? GetSnapshot(service, cache);
		}

		return snapshot;
	}

	private static async Task<FullRunSimulationStateSnapshot> AutoAdvancePendingWaitStatesAsync(
		FullRunTrainingEnvService service,
		RequestStateCache cache,
		FullRunSimulationStateSnapshot snapshot,
		int maxAutoAdvance)
	{
		for (int attempt = 0; attempt < maxAutoAdvance; attempt++)
		{
			if (snapshot.IsTerminal || snapshot.StateType == "game_over")
			{
				break;
			}

			if (snapshot.LegalActions.Count > 0)
			{
				break;
			}

			FullRunSimulationDiagnostics.Increment("step.auto_wait");
			FullRunSimulationStepResult waitResult = await service.StepAsync(new FullRunSimulationActionRequest { Action = "wait" });
			snapshot = waitResult.State ?? GetSnapshot(service, cache);
		}

		return snapshot;
	}
}
