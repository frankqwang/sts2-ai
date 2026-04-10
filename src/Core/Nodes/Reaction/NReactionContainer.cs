using Godot;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Multiplayer.Game;

namespace MegaCrit.Sts2.Core.Nodes.Reaction;

public partial class NReactionContainer : Control
{
	private ReactionSynchronizer? _synchronizer;

	public bool InMultiplayer
	{
		get
		{
			if (_synchronizer != null)
			{
				return _synchronizer.NetService.Type != NetGameType.Singleplayer;
			}
			return false;
		}
	}

	public void InitializeNetworking(INetGameService netService)
	{
		if (_synchronizer != null)
		{
			DeinitializeNetworking();
		}
		_synchronizer = new ReactionSynchronizer(netService, this);
		_synchronizer.NetService.Disconnected += NetServiceDisconnected;
	}

	private void NetServiceDisconnected(NetErrorInfo _)
	{
		DeinitializeNetworking();
	}

	public void DeinitializeNetworking()
	{
		if (_synchronizer != null)
		{
			_synchronizer.NetService.Disconnected -= NetServiceDisconnected;
			_synchronizer.Dispose();
			_synchronizer = null;
		}
	}

	public override void _ExitTree()
	{
		DeinitializeNetworking();
	}

	public void DoLocalReaction(Texture2D tex, Vector2 position)
	{
		NReaction nReaction = NReaction.Create(tex);
		this.AddChildSafely(nReaction);
		nReaction.GlobalPosition = position - nReaction.Size / 2f;
		nReaction.BeginAnim();
		_synchronizer?.SendLocalReaction(nReaction.Type, position);
	}

	public void DoRemoteReaction(ReactionType type, Vector2 position)
	{
		NReaction nReaction = NReaction.Create(type);
		this.AddChildSafely(nReaction);
		nReaction.GlobalPosition = position - nReaction.Size / 2f;
		nReaction.BeginAnim();
	}
}
