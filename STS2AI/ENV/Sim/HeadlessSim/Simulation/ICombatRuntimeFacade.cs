using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Training;

namespace MegaCrit.Sts2.Core.Simulation;

public interface ICombatRuntimeFacade
{
	bool IsActive { get; }

	bool IsPureSimulator { get; }

	bool? LastCombatWasVictory { get; }

	int EpisodeNumber { get; }

	string? CurrentSeed { get; }

	string? CurrentCharacterId { get; }

	string? CurrentEncounterId { get; }

	int CurrentAscensionLevel { get; }

	Task<CombatTrainingStateSnapshot> ResetAsync(CombatTrainingResetRequest? request = null);

	CombatTrainingStateSnapshot GetState();

	Task<CombatTrainingStepResult> StepAsync(CombatTrainingActionRequest action);
}
