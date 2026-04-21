using System;
using System.Buffers.Binary;
using System.IO;
using System.IO.Pipes;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Simulation;
using STS2AI.Bridge;

namespace HeadlessSim;

// Pipe server + named-pipe transport helpers.
// See Program.Proto.cs / Program.Json.cs for the per-protocol request handlers
// that this layer dispatches to.
internal static partial class Program
{
	private static async Task RunPipeServerAsync(FullRunTrainingEnvService service, HostOptions options)
	{
		using CancellationTokenSource cts = new CancellationTokenSource();
		Console.CancelKeyPress += (_, eventArgs) =>
		{
			eventArgs.Cancel = true;
			cts.Cancel();
		};

		PipeSessionManager sessions = new PipeSessionManager();
		while (!cts.IsCancellationRequested)
		{
			NamedPipeServerStream? server = null;
			try
			{
				server = new NamedPipeServerStream(
					options.PipeName,
					PipeDirection.InOut,
					NamedPipeServerStream.MaxAllowedServerInstances,
					PipeTransmissionMode.Byte,
					PipeOptions.Asynchronous);

				await server.WaitForConnectionAsync(cts.Token);
				NamedPipeServerStream connectedServer = server;
				_ = Task.Run(
					() => HandlePipeConnectionAsync(service, connectedServer, sessions, options, cts.Token),
					cts.Token);
				server = null;
			}
			catch (OperationCanceledException)
			{
				break;
			}
			catch (Exception ex)
			{
				Console.Error.WriteLine($"HeadlessSim: pipe listener error: {ex}");
				await Task.Delay(100, cts.Token);
			}
			finally
			{
				server?.Dispose();
			}
		}
	}

	private static async Task HandlePipeConnectionAsync(
		FullRunTrainingEnvService service,
		NamedPipeServerStream pipe,
		PipeSessionManager sessions,
		HostOptions options,
		CancellationToken cancellationToken)
	{
		long sessionId = sessions.TryAcquire();
		if (sessionId < 0)
		{
			using (pipe)
			{
				if (options.Protocol == HostProtocol.Proto)
				{
					await WritePipeMessageAsync(
						pipe,
						ProtoStateBuilder.BuildErrorResponse(
							PipeMethod.Handshake,
							PipeStatus.ProtocolError,
							"simulator_busy",
							"The simulator runtime is already owned by another active pipe session."),
						cancellationToken);
				}
				else
				{
					await WritePipeMessageAsync(
						pipe,
						SerializePipeError("simulator_busy", "The simulator runtime is already owned by another active pipe session."),
						cancellationToken);
				}
			}
			return;
		}

		try
		{
			using (pipe)
			{
				if (options.Protocol == HostProtocol.Proto)
				{
					await WritePipeMessageAsync(pipe, ProtoStateBuilder.BuildHandshakeResponse(), cancellationToken);
				}
				else
				{
					await WritePipeMessageAsync(pipe, JsonSerializer.Serialize(new { ok = true }, JsonOptions), cancellationToken);
				}

				while (pipe.IsConnected && !cancellationToken.IsCancellationRequested)
				{
					byte[]? requestBytes = await ReadPipeMessageBytesAsync(pipe, options.ReadTimeout, cancellationToken);
					if (requestBytes == null)
					{
						break;
					}

					if (options.Protocol == HostProtocol.Proto)
					{
						byte[] responseBytes;
						try
						{
							using CancellationTokenSource requestCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
							requestCts.CancelAfter(options.RequestTimeout);
							responseBytes = await ProcessProtoRequestAsync(service, requestBytes).WaitAsync(requestCts.Token);
						}
						catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
						{
							responseBytes = ProtoStateBuilder.BuildErrorResponse(
								SafeParseMethod(requestBytes),
								PipeStatus.ProtocolError,
								"request_timeout",
								$"Request processing timed out after {options.RequestTimeout.TotalSeconds:F0}s");
						}
						catch (Exception ex)
						{
							Console.Error.WriteLine($"HeadlessSim: proto request error method={SafeParseMethod(requestBytes)}: {ex}");
							responseBytes = ProtoStateBuilder.BuildErrorResponse(
								SafeParseMethod(requestBytes),
								GetProtoPipeErrorStatus(ex),
								GetStructuredErrorCode(ex) ?? "internal_error",
								ex.Message);
						}

						await WritePipeMessageAsync(pipe, responseBytes, cancellationToken);
					}
					else
					{
						string requestJson = Encoding.UTF8.GetString(requestBytes);
						string responseJson;
						try
						{
							using CancellationTokenSource requestCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
							requestCts.CancelAfter(options.RequestTimeout);
							responseJson = await ProcessPipeRequestAsync(service, requestJson).WaitAsync(requestCts.Token);
						}
						catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
						{
							responseJson = SerializePipeError(
								"request_timeout",
								$"Request processing timed out after {options.RequestTimeout.TotalSeconds:F0}s");
						}
						catch (Exception ex)
						{
							Console.Error.WriteLine($"HeadlessSim: json request error request={requestJson}: {ex}");
							responseJson = SerializePipeError(GetStructuredErrorCode(ex) ?? "internal_error", ex.Message);
						}

						await WritePipeMessageAsync(pipe, responseJson, cancellationToken);
					}
				}
			}
		}
		catch (IOException)
		{
		}
		catch (OperationCanceledException)
		{
		}
		catch (Exception ex)
		{
			Console.Error.WriteLine($"HeadlessSim: pipe connection error: {ex}");
		}
		finally
		{
			sessions.Release(sessionId);
		}
	}

	private static async Task<string?> ReadPipeMessageAsync(Stream stream, TimeSpan readTimeout, CancellationToken cancellationToken)
	{
		byte[]? payload = await ReadPipeMessageBytesAsync(stream, readTimeout, cancellationToken);
		return payload == null ? null : Encoding.UTF8.GetString(payload);
	}

	private static async Task<byte[]?> ReadPipeMessageBytesAsync(Stream stream, TimeSpan readTimeout, CancellationToken cancellationToken)
	{
		using CancellationTokenSource readCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
		readCts.CancelAfter(readTimeout);

		byte[] lenBuffer = new byte[4];
		int lenRead;
		try
		{
			lenRead = await ReadExactAsync(stream, lenBuffer, readCts.Token);
		}
		catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
		{
			return null;
		}

		if (lenRead == 0)
		{
			return null;
		}

		if (lenRead < 4)
		{
			throw new EndOfStreamException("Incomplete pipe length prefix.");
		}

		int messageLength = BinaryPrimitives.ReadInt32LittleEndian(lenBuffer);
		if (messageLength <= 0 || messageLength > 10_000_000)
		{
			throw new InvalidOperationException($"Invalid pipe message length: {messageLength}");
		}

		byte[] messageBuffer = new byte[messageLength];
		int messageRead = await ReadExactAsync(stream, messageBuffer, readCts.Token);
		if (messageRead < messageLength)
		{
			throw new EndOfStreamException("Incomplete pipe payload.");
		}

		return messageBuffer;
	}

	private static async Task<int> ReadExactAsync(Stream stream, byte[] buffer, CancellationToken cancellationToken)
	{
		int offset = 0;
		while (offset < buffer.Length)
		{
			int read = await stream.ReadAsync(buffer.AsMemory(offset, buffer.Length - offset), cancellationToken);
			if (read == 0)
			{
				return offset;
			}

			offset += read;
		}

		return offset;
	}

	private static async Task WritePipeMessageAsync(Stream stream, string payload, CancellationToken cancellationToken)
	{
		byte[] body = Encoding.UTF8.GetBytes(payload);
		await WritePipeMessageAsync(stream, body, cancellationToken);
	}

	private static async Task WritePipeMessageAsync(Stream stream, byte[] body, CancellationToken cancellationToken)
	{
		byte[] prefix = new byte[4];
		BinaryPrimitives.WriteInt32LittleEndian(prefix, body.Length);
		await stream.WriteAsync(prefix, cancellationToken);
		await stream.WriteAsync(body, cancellationToken);
		await stream.FlushAsync(cancellationToken);
	}

	private sealed class PipeSessionManager
	{
		private readonly object _sync = new object();
		private long _nextSessionId;
		private long? _activeSessionId;

		public long TryAcquire()
		{
			lock (_sync)
			{
				if (_activeSessionId.HasValue)
				{
					return -1;
				}

				_nextSessionId++;
				_activeSessionId = _nextSessionId;
				return _nextSessionId;
			}
		}

		public void Release(long sessionId)
		{
			lock (_sync)
			{
				if (_activeSessionId == sessionId)
				{
					_activeSessionId = null;
				}
			}
		}
	}
}
