using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace MegaCrit.Sts2.Core.Simulation;

public static class FullRunSimulationDiagnostics
{
	private sealed class TimingAggregate
	{
		public long Count;

		public double TotalMs;

		public double MaxMs;
	}

	private static readonly object Sync = new();
	private static readonly Dictionary<string, long> Counters = new(StringComparer.Ordinal);
	private static readonly Dictionary<string, TimingAggregate> Timings = new(StringComparer.Ordinal);

	public static IDisposable Measure(string name)
	{
		long start = Stopwatch.GetTimestamp();
		return new MeasureScope(name, start);
	}

	public static void RecordTiming(string name, double elapsedMs)
	{
		lock (Sync)
		{
			if (!Timings.TryGetValue(name, out TimingAggregate? aggregate))
			{
				aggregate = new TimingAggregate();
				Timings[name] = aggregate;
			}

			aggregate.Count++;
			aggregate.TotalMs += elapsedMs;
			if (elapsedMs > aggregate.MaxMs)
			{
				aggregate.MaxMs = elapsedMs;
			}
		}
	}

	public static void Increment(string name, long delta = 1)
	{
		lock (Sync)
		{
			Counters.TryGetValue(name, out long current);
			Counters[name] = current + delta;
		}
	}

	public static Dictionary<string, object?> Snapshot()
	{
		lock (Sync)
		{
			return new Dictionary<string, object?>
			{
				["counters"] = Counters.OrderBy(static entry => entry.Key)
					.ToDictionary(static entry => entry.Key, static entry => (object?)entry.Value, StringComparer.Ordinal),
				["timings"] = Timings.OrderBy(static entry => entry.Key)
					.ToDictionary(
						static entry => entry.Key,
						static entry => (object?)new Dictionary<string, object?>
						{
							["count"] = entry.Value.Count,
							["total_ms"] = entry.Value.TotalMs,
							["avg_ms"] = entry.Value.Count == 0 ? 0.0 : entry.Value.TotalMs / entry.Value.Count,
							["max_ms"] = entry.Value.MaxMs
						},
						StringComparer.Ordinal)
			};
		}
	}

	public static void Reset()
	{
		lock (Sync)
		{
			Counters.Clear();
			Timings.Clear();
		}
	}

	private sealed class MeasureScope : IDisposable
	{
		private readonly string _name;
		private readonly long _startTimestamp;
		private bool _disposed;

		public MeasureScope(string name, long startTimestamp)
		{
			_name = name;
			_startTimestamp = startTimestamp;
		}

		public void Dispose()
		{
			if (_disposed)
			{
				return;
			}

			_disposed = true;
			double elapsedMs = (Stopwatch.GetTimestamp() - _startTimestamp) * 1000.0 / Stopwatch.Frequency;
			RecordTiming(_name, elapsedMs);
		}
	}
}
