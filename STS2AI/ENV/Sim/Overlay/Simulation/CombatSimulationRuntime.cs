using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Combat;

namespace MegaCrit.Sts2.Core.Simulation;

public static class CombatSimulationRuntime
{
	private static readonly object _sync = new object();

	private static int _pureCombatSimulatorDepth;

	private static readonly Stack<ICombatPresentation> _presentationStack = new Stack<ICombatPresentation>();

	public static ICombatPresentation Presentation { get; private set; } = NullCombatPresentation.Instance;

	public static bool IsPureCombatSimulator
	{
		get
		{
			lock (_sync)
			{
				return _pureCombatSimulatorDepth > 0;
			}
		}
	}

	public static ISimulatorClock Clock
	{
		get
		{
			return IsPureCombatSimulator ? ImmediateSimulatorClock.Instance : GodotSimulatorClock.Instance;
		}
	}

	public static IDisposable EnterPureCombatSimulator(ICombatPresentation? presentation = null)
	{
		lock (_sync)
		{
			_presentationStack.Push(Presentation);
			_pureCombatSimulatorDepth++;
			Presentation = presentation ?? NullCombatPresentation.Instance;
		}
		return new Scope();
	}

	private static void ExitPureCombatSimulator()
	{
		lock (_sync)
		{
			if (_pureCombatSimulatorDepth > 0)
			{
				_pureCombatSimulatorDepth--;
			}
			Presentation = _presentationStack.Count > 0 ? _presentationStack.Pop() : NullCombatPresentation.Instance;
		}
	}

	private sealed class Scope : IDisposable
	{
		private bool _disposed;

		public void Dispose()
		{
			if (_disposed)
			{
				return;
			}
			_disposed = true;
			ExitPureCombatSimulator();
		}
	}
}

public interface ICombatPresentation
{
	void OnCombatStateChanged(CombatState state);
}

public sealed class NullCombatPresentation : ICombatPresentation
{
	public static NullCombatPresentation Instance { get; } = new NullCombatPresentation();

	private NullCombatPresentation()
	{
	}

	public void OnCombatStateChanged(CombatState state)
	{
	}
}

public interface ISimulatorClock
{
	Task YieldAsync();
}

public sealed class ImmediateSimulatorClock : ISimulatorClock
{
	public static ImmediateSimulatorClock Instance { get; } = new ImmediateSimulatorClock();

	private ImmediateSimulatorClock()
	{
	}

	public Task YieldAsync()
	{
		return Task.CompletedTask;
	}
}

public sealed class GodotSimulatorClock : ISimulatorClock
{
	public static GodotSimulatorClock Instance { get; } = new GodotSimulatorClock();

	private GodotSimulatorClock()
	{
	}

	public async Task YieldAsync()
	{
		if (Engine.GetMainLoop() != null)
		{
			await Engine.GetMainLoop().ToSignal(Engine.GetMainLoop(), SceneTree.SignalName.ProcessFrame);
		}
	}
}
