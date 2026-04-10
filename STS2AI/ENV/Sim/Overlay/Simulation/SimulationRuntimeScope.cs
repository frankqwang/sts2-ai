using System;

namespace MegaCrit.Sts2.Core.Simulation;

public sealed class SimulationRuntimeScope : IDisposable
{
	private readonly IDisposable _scope;

	public SimulationRuntimeScope(ICombatPresentation? presentation = null)
	{
		_scope = CombatSimulationRuntime.EnterPureCombatSimulator(presentation);
	}

	public void Dispose()
	{
		_scope.Dispose();
	}
}
