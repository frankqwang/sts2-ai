using System.Threading.Tasks;
using STS2AI.Bridge;
using STS2AI.Bridge.Runtime;

namespace STS2_MCP;

public static partial class McpMod
{
    private sealed class SpectatorBridgeRuntime : BridgeRuntimeBase
    {
        public override Task<BridgeResponseEnvelope> StateAsync(BridgeRequestEnvelope request) =>
            Task.FromResult(BuildBridgeStateResponse(BridgeMethod.State));

        public override Task<BridgeResponseEnvelope> ResetAsync(BridgeRequestEnvelope request) =>
            Task.FromResult(ProcessBridgeReset(request));

        public override Task<BridgeResponseEnvelope> ActAsync(BridgeRequestEnvelope request) =>
            Task.FromResult(ProcessBridgeAct(request, BridgeMethod.Act));

        public override Task<BridgeResponseEnvelope> CombatResetAsync(BridgeRequestEnvelope request) =>
            Task.FromResult(ProcessBridgeReset(request, forceCombatReset: true));

        public override Task<BridgeResponseEnvelope> CombatActAsync(BridgeRequestEnvelope request) =>
            Task.FromResult(ProcessBridgeAct(request, BridgeMethod.CombatAct));

        public override Task<BridgeResponseEnvelope> CombatStateAsync(BridgeRequestEnvelope request) =>
            Task.FromResult(BuildBridgeStateResponse(BridgeMethod.CombatState));

        public override Task<BridgeResponseEnvelope> SkipCombatAsync(BridgeRequestEnvelope request) =>
            Task.FromResult(ProcessBridgeSkipCombat());
    }
}
