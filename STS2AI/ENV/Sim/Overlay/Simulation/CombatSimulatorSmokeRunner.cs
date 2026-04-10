using System;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Training;

namespace MegaCrit.Sts2.Core.Simulation;

public sealed class CombatSimulatorSmokeRunner
{
	public async Task<int> StartAsync(NGame game)
	{
		if (game == null)
		{
			throw new ArgumentNullException(nameof(game));
		}
		try
		{
			CombatSimulationTrace.Reset();
			CombatSimulationTrace.Write("smoke.start");
			Log.Info("[CombatSimSmoke] Starting pure simulator smoke run");
			CombatSimulationTrace.Write("smoke.before_reset");
			CombatTrainingStateSnapshot state = await CombatTrainingEnvService.Instance.ResetAsync(new CombatTrainingResetRequest
			{
				CharacterId = CombatSimulationMode.CharacterId,
				EncounterId = CombatSimulationMode.EncounterId,
				Seed = CombatSimulationMode.Seed,
				AscensionLevel = CombatSimulationMode.AscensionLevel
			});
			CombatSimulationTrace.Write($"smoke.after_reset pure={state.IsPureSimulator} encounter={state.EncounterId} seed={state.Seed}");
			Log.Info($"[CombatSimSmoke] Reset complete. pure={state.IsPureSimulator} character={state.CharacterId} encounter={state.EncounterId} seed={state.Seed} ascension={state.AscensionLevel}");
			int steps = 0;
			while (!state.IsEpisodeDone && steps < CombatSimulationMode.StepBudget)
			{
				CombatTrainingActionRequest action = BuildAction(state);
				CombatSimulationTrace.Write($"smoke.step.{steps}.action={action.Type}");
				CombatTrainingStepResult result = await CombatTrainingEnvService.Instance.StepAsync(action);
				if (!result.Accepted)
				{
					CombatSimulationTrace.Write($"smoke.step.{steps}.rejected={result.Error}");
					Log.Error($"[CombatSimSmoke] Action rejected at step {steps}: {result.Error}");
					return 3;
				}
				if (result.State == null)
				{
					CombatSimulationTrace.Write($"smoke.step.{steps}.missing_state");
					Log.Error($"[CombatSimSmoke] Missing state snapshot at step {steps}");
					return 4;
				}
				state = result.State;
				steps++;
			}
			if (!state.IsEpisodeDone)
			{
				CombatSimulationTrace.Write($"smoke.step_budget_exhausted steps={steps}");
				Log.Error($"[CombatSimSmoke] Step budget exhausted. steps={steps} active={state.IsCombatActive} selection={state.IsHandSelectionActive}");
				return 5;
			}
			CombatSimulationTrace.Write($"smoke.done steps={steps} victory={state.Victory}");
			Log.Info($"[CombatSimSmoke] Finished. steps={steps} done={state.IsEpisodeDone} victory={state.Victory} active={state.IsCombatActive} selection={state.IsHandSelectionActive}");
			return 0;
		}
		catch (Exception ex)
		{
			CombatSimulationTrace.Write($"smoke.exception={ex.GetType().Name}:{ex.Message}");
			Log.Error($"[CombatSimSmoke] Failed: {ex}");
			return 1;
		}
	}

	private static CombatTrainingActionRequest BuildAction(CombatTrainingStateSnapshot state)
	{
		return CombatSimulatorAutoplay.BuildAction(state);
	}
}
