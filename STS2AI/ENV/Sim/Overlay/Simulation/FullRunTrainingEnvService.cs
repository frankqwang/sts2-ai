using System;
using System.Threading;
using System.Threading.Tasks;

namespace MegaCrit.Sts2.Core.Simulation;

public sealed class FullRunTrainingEnvService
{
	public static FullRunTrainingEnvService Instance { get; } = new FullRunTrainingEnvService();

	private static readonly IFullRunRuntimeFacade SimulatorRuntime = new FullRunSimulatorRuntimeFacade();
	private static int _standaloneScopeDepth;
	private readonly SemaphoreSlim _operationGate = new SemaphoreSlim(1, 1);

	private FullRunTrainingEnvService()
	{
	}

	public Task<FullRunSimulationStateSnapshot> ResetAsync(FullRunSimulationResetRequest? request = null)
	{
		return ExecuteSerializedAsync(() => SimulatorRuntime.ResetAsync(request));
	}

	public FullRunSimulationStateSnapshot GetState()
	{
		return ExecuteSerialized(SimulatorRuntime.GetState);
	}

	public Task<FullRunSimulationStepResult> StepAsync(FullRunSimulationActionRequest action)
	{
		return ExecuteSerializedAsync(() => SimulatorRuntime.StepAsync(action));
	}

	public Task<FullRunSimulationBatchStepResult> BatchStepAsync(System.Collections.Generic.IReadOnlyList<FullRunSimulationActionRequest> actions)
	{
		return ExecuteSerializedAsync(() => ((FullRunSimulatorRuntimeFacade)SimulatorRuntime).BatchStepAsync(actions));
	}

	// ------------------------------------------------------------------
	// MCTS State Snapshot API
	// ------------------------------------------------------------------

	/// <summary>Save current game state. Returns a state_id for later restore.</summary>
	public string SaveState()
	{
		return ExecuteSerialized(() => ((FullRunSimulatorRuntimeFacade)SimulatorRuntime).SaveState());
	}

	/// <summary>Restore game to a previously saved state.</summary>
	public Task<FullRunSimulationStateSnapshot> LoadState(string stateId)
	{
		return ExecuteSerializedAsync(() => ((FullRunSimulatorRuntimeFacade)SimulatorRuntime).LoadState(stateId));
	}

	/// <summary>Export a saved snapshot (or current state) to a JSON file.</summary>
	public string ExportStateToFile(string path, string? stateId = null)
	{
		return ExecuteSerialized(() => ((FullRunSimulatorRuntimeFacade)SimulatorRuntime).ExportStateToFile(path, stateId));
	}

	/// <summary>Load a previously exported snapshot JSON file.</summary>
	public Task<FullRunSimulationStateSnapshot> LoadStateFromFile(string path)
	{
		return ExecuteSerializedAsync(() => ((FullRunSimulatorRuntimeFacade)SimulatorRuntime).LoadStateFromFile(path));
	}

	/// <summary>Delete a saved state snapshot.</summary>
	public bool DeleteState(string stateId)
	{
		return ExecuteSerialized(() => ((FullRunSimulatorRuntimeFacade)SimulatorRuntime).DeleteState(stateId));
	}

	/// <summary>Clear all saved state snapshots.</summary>
	public void ClearStateCache()
	{
		ExecuteSerialized(() => ((FullRunSimulatorRuntimeFacade)SimulatorRuntime).ClearStateCache());
	}

	/// <summary>Number of cached state snapshots.</summary>
	public int StateCacheCount
	{
		get
		{
			return ExecuteSerialized(() => ((FullRunSimulatorRuntimeFacade)SimulatorRuntime).StateCacheCount);
		}
	}

	/// <summary>
	/// Whether this service wraps a pure simulator (always true for FullRunSimulatorRuntimeFacade).
	/// Off-main-thread access is allowed, but operations are serialized so the runtime behaves
	/// like a single deterministic simulator instance.
	/// </summary>
	public bool IsPureSimulator => SimulatorRuntime.IsPureSimulator;

	/// <summary>
	/// Allows a standalone host process to use the simulator without relying on
	/// Godot's command-line mode flags.
	/// </summary>
	public static IDisposable EnterStandaloneMode()
	{
		Interlocked.Increment(ref _standaloneScopeDepth);
		return new StandaloneScope();
	}

	/// <summary>Max time to wait for the operation gate (seconds).
	/// If a previous operation is stuck, this prevents permanent deadlock.</summary>
	private const int GateTimeoutSeconds = 30;

	private async Task<T> ExecuteSerializedAsync<T>(Func<Task<T>> operation)
	{
		EnsureEnvIsAvailable();
		if (!await _operationGate.WaitAsync(TimeSpan.FromSeconds(GateTimeoutSeconds)).ConfigureAwait(false))
		{
			throw new TimeoutException(
				$"FullRunTrainingEnvService operation gate timed out after {GateTimeoutSeconds}s. " +
				"A previous operation may be stuck.");
		}
		try
		{
			return await operation().ConfigureAwait(false);
		}
		finally
		{
			_operationGate.Release();
		}
	}

	private T ExecuteSerialized<T>(Func<T> operation)
	{
		EnsureEnvIsAvailable();
		if (!_operationGate.Wait(TimeSpan.FromSeconds(GateTimeoutSeconds)))
		{
			throw new TimeoutException(
				$"FullRunTrainingEnvService operation gate timed out after {GateTimeoutSeconds}s.");
		}
		try
		{
			return operation();
		}
		finally
		{
			_operationGate.Release();
		}
	}

	private void ExecuteSerialized(Action operation)
	{
		EnsureEnvIsAvailable();
		if (!_operationGate.Wait(TimeSpan.FromSeconds(GateTimeoutSeconds)))
		{
			throw new TimeoutException(
				$"FullRunTrainingEnvService operation gate timed out after {GateTimeoutSeconds}s.");
		}
		try
		{
			operation();
		}
		finally
		{
			_operationGate.Release();
		}
	}

	private static void EnsureEnvIsAvailable()
	{
		if (Volatile.Read(ref _standaloneScopeDepth) <= 0 && !FullRunSimulationMode.IsAnyActive)
		{
			throw new InvalidOperationException("FullRunTrainingEnvService requires --full-run-sim-server or --full-run-sim-smoke.");
		}
		// Pure-sim facade (FullRunSimulatorRuntimeFacade.IsPureSimulator == true): all game logic
		// uses TestMode.IsOn + ImmediateSimulatorClock, so off-main-thread access is allowed.
		// Calls are still serialized by FullRunTrainingEnvService to prevent overlapping runtime
		// mutations against the shared simulator instance.
		if (SimulatorRuntime.IsPureSimulator)
		{
			return;
		}
		if ((FullRunSimulationMode.IsServerActive || FullRunSimulationMode.IsSmokeActive)
			&& !Nodes.NGame.IsMainThread())
		{
			throw new InvalidOperationException("FullRunTrainingEnvService must be called from the main game thread.");
		}
	}

	private sealed class StandaloneScope : IDisposable
	{
		private int _disposed;

		public void Dispose()
		{
			if (Interlocked.Exchange(ref _disposed, 1) == 0)
			{
				Interlocked.Decrement(ref _standaloneScopeDepth);
			}
		}
	}
}
