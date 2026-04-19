using System;

namespace MegaCrit.Sts2.Core.Simulation;

public sealed class FullRunSimulationRuntimeException : InvalidOperationException
{
	public string ErrorCode { get; }

	public FullRunSimulationRuntimeException(string errorCode, string message)
		: base(message)
	{
		ErrorCode = errorCode;
	}
}
