using System.Collections.Concurrent;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;

namespace MegaCrit.Sts2.Core.Nodes;

public partial class NAssetLoader : Node
{
	private static NAssetLoader? _instance;

	private readonly ConcurrentQueue<AssetLoadingSession?> _sessions = new ConcurrentQueue<AssetLoadingSession>();

	private AssetLoadingSession? _currentSession;

	public static NAssetLoader Instance => _instance ?? new NAssetLoader();

	public override void _Ready()
	{
		if (_instance == null)
		{
			_instance = this;
		}
	}

	public Task<bool> LoadInTheBackground(AssetLoadingSession session)
	{
		_sessions.Enqueue(session);
		SetProcess(enable: true);
		return session.Task;
	}

	public override void _Process(double delta)
	{
		if (_currentSession == null || _currentSession.IsCompleted)
		{
			if (_sessions.TryDequeue(out AssetLoadingSession result))
			{
				_currentSession = result;
			}
			else
			{
				SetProcess(enable: false);
			}
		}
		else
		{
			_currentSession.Process();
		}
	}
}
