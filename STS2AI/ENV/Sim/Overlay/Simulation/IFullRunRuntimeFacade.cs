using System.Threading.Tasks;

namespace MegaCrit.Sts2.Core.Simulation;

public interface IFullRunRuntimeFacade
{
	bool IsActive { get; }

	bool IsPureSimulator { get; }

	int EpisodeNumber { get; }

	string? CurrentSeed { get; }

	string? CurrentCharacterId { get; }

	int CurrentAscensionLevel { get; }

	Task<FullRunSimulationStateSnapshot> ResetAsync(FullRunSimulationResetRequest? request = null);

	FullRunSimulationStateSnapshot GetState();

	Task<FullRunSimulationStepResult> StepAsync(FullRunSimulationActionRequest action);
}
