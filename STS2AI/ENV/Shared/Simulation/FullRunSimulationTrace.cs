using System;
using System.IO;
using System.Text;

namespace MegaCrit.Sts2.Core.Simulation;

internal static class FullRunSimulationTrace
{
	private static readonly string TraceDir = Path.Combine("artifacts", "full_run_sim_smoke");

	private static readonly string TraceFile = Path.Combine(TraceDir, "latest.log");

	private static readonly object Sync = new object();

	public static void Reset()
	{
		lock (Sync)
		{
			Directory.CreateDirectory(TraceDir);
			using FileStream stream = new FileStream(TraceFile, FileMode.Create, FileAccess.Write, FileShare.ReadWrite);
			using StreamWriter writer = new StreamWriter(stream, Encoding.UTF8);
			writer.Write(string.Empty);
		}
	}

	public static void Write(string message)
	{
		lock (Sync)
		{
			Directory.CreateDirectory(TraceDir);
			using FileStream stream = new FileStream(TraceFile, FileMode.Append, FileAccess.Write, FileShare.ReadWrite);
			using StreamWriter writer = new StreamWriter(stream, Encoding.UTF8);
			writer.WriteLine($"{DateTime.UtcNow:O} {message}");
		}
	}
}
