using System;

namespace STS2_MCP;

internal sealed class McpApiException : InvalidOperationException
{
    internal McpApiException(string errorCode, string message)
        : base(message)
    {
        ErrorCode = errorCode;
    }

    internal string ErrorCode { get; }
}
