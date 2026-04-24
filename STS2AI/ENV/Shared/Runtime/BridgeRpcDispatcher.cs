using System.Threading.Tasks;
using STS2AI.Bridge;

namespace STS2AI.Bridge.Runtime;

public abstract class BridgeRuntimeBase
{
	public abstract Task<BridgeResponseEnvelope> ResetAsync(BridgeRequestEnvelope request);
	public abstract Task<BridgeResponseEnvelope> StateAsync(BridgeRequestEnvelope request);
	public abstract Task<BridgeResponseEnvelope> ActAsync(BridgeRequestEnvelope request);
	public abstract Task<BridgeResponseEnvelope> CombatResetAsync(BridgeRequestEnvelope request);
	public abstract Task<BridgeResponseEnvelope> CombatActAsync(BridgeRequestEnvelope request);
	public abstract Task<BridgeResponseEnvelope> CombatStateAsync(BridgeRequestEnvelope request);

	public virtual Task<BridgeResponseEnvelope> BatchActAsync(BridgeRequestEnvelope request) => UnsupportedAsync(request.Method);
	public virtual Task<BridgeResponseEnvelope> SaveStateAsync(BridgeRequestEnvelope request) => UnsupportedAsync(request.Method);
	public virtual Task<BridgeResponseEnvelope> SaveSearchStateAsync(BridgeRequestEnvelope request) => UnsupportedAsync(request.Method);
	public virtual Task<BridgeResponseEnvelope> ExportStateAsync(BridgeRequestEnvelope request) => UnsupportedAsync(request.Method);
	public virtual Task<BridgeResponseEnvelope> LoadStateAsync(BridgeRequestEnvelope request) => UnsupportedAsync(request.Method);
	public virtual Task<BridgeResponseEnvelope> ImportStateAsync(BridgeRequestEnvelope request) => UnsupportedAsync(request.Method);
	public virtual Task<BridgeResponseEnvelope> DeleteStateAsync(BridgeRequestEnvelope request) => UnsupportedAsync(request.Method);
	public virtual Task<BridgeResponseEnvelope> PerfStatsAsync(BridgeRequestEnvelope request) => UnsupportedAsync(request.Method);
	public virtual Task<BridgeResponseEnvelope> ResetPerfStatsAsync(BridgeRequestEnvelope request) => UnsupportedAsync(request.Method);
	public virtual Task<BridgeResponseEnvelope> StepLocalPolicyAsync(BridgeRequestEnvelope request) => UnsupportedAsync(request.Method);
	public virtual Task<BridgeResponseEnvelope> LoadOrtModelAsync(BridgeRequestEnvelope request) => UnsupportedAsync(request.Method);
	public virtual Task<BridgeResponseEnvelope> RunCombatLocalAsync(BridgeRequestEnvelope request) => UnsupportedAsync(request.Method);
	public virtual Task<BridgeResponseEnvelope> SkipCombatAsync(BridgeRequestEnvelope request) => UnsupportedAsync(request.Method);
	public virtual Task<BridgeResponseEnvelope> SearchCombatMctsAsync(BridgeRequestEnvelope request) => UnsupportedAsync(request.Method);

	protected static Task<BridgeResponseEnvelope> UnsupportedAsync(BridgeMethod method)
	{
		return Task.FromResult(new BridgeResponseEnvelope
		{
			Method = method,
			Status = BridgeStatus.UnsupportedBackend,
			Error = new BridgeError
			{
				ErrorCode = "unsupported_backend",
				ErrorMessage = $"{method} is unsupported by this backend.",
			},
		});
	}

	internal static BridgeResponseEnvelope ProtocolError(BridgeMethod method, string code, string message)
	{
		return new BridgeResponseEnvelope
		{
			Method = method,
			Status = BridgeStatus.ProtocolError,
			Error = new BridgeError
			{
				ErrorCode = code,
				ErrorMessage = message,
			},
		};
	}
}

public static class BridgeRpcDispatcher
{
	public static Task<BridgeResponseEnvelope> DispatchAsync(BridgeRuntimeBase runtime, BridgeRequestEnvelope request)
	{
		return request.Method switch
		{
			BridgeMethod.Reset => runtime.ResetAsync(request),
			BridgeMethod.State => runtime.StateAsync(request),
			BridgeMethod.Act => runtime.ActAsync(request),
			BridgeMethod.BatchAct => runtime.BatchActAsync(request),
			BridgeMethod.SaveState => runtime.SaveStateAsync(request),
			BridgeMethod.SaveSearchState => runtime.SaveSearchStateAsync(request),
			BridgeMethod.ExportState => runtime.ExportStateAsync(request),
			BridgeMethod.LoadState => runtime.LoadStateAsync(request),
			BridgeMethod.ImportState => runtime.ImportStateAsync(request),
			BridgeMethod.DeleteState => runtime.DeleteStateAsync(request),
			BridgeMethod.PerfStats => runtime.PerfStatsAsync(request),
			BridgeMethod.ResetPerfStats => runtime.ResetPerfStatsAsync(request),
			BridgeMethod.StepLocalPolicy => runtime.StepLocalPolicyAsync(request),
			BridgeMethod.LoadOrtModel => runtime.LoadOrtModelAsync(request),
			BridgeMethod.RunCombatLocal => runtime.RunCombatLocalAsync(request),
			BridgeMethod.SkipCombat => runtime.SkipCombatAsync(request),
			BridgeMethod.SearchCombatMcts => runtime.SearchCombatMctsAsync(request),
			BridgeMethod.CombatReset => runtime.CombatResetAsync(request),
			BridgeMethod.CombatAct => runtime.CombatActAsync(request),
			BridgeMethod.CombatState => runtime.CombatStateAsync(request),
			_ => Task.FromResult(BridgeRuntimeBase.ProtocolError(
				request.Method,
				"unknown_method",
				$"Unknown method: {request.Method}")),
		};
	}
}
