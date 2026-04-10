using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Multiplayer.Game.Lobby;
using MegaCrit.Sts2.Core.Saves.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Multiplayer;

public partial class NRemoteLoadLobbyPlayerContainer : Control
{
	private readonly List<NRemoteLobbyPlayer> _nodes = new List<NRemoteLobbyPlayer>();

	private LoadRunLobby? _lobby;

	private MegaLabel? _othersLabel;

	private Control _container;

	public override void _Ready()
	{
		_othersLabel = GetNodeOrNull<MegaLabel>("OthersLabel");
		_container = GetNode<Control>("Container");
	}

	public void Initialize(LoadRunLobby runLobby, bool displayLocalPlayer)
	{
		_lobby = runLobby;
		if (_othersLabel != null)
		{
			LocString locString = new LocString("main_menu_ui", "MULTIPLAYER_LOAD_MENU.OTHERS");
			locString.Add("others", runLobby.Run.Players.Count - 1);
			_othersLabel.Text = locString.GetFormattedText();
		}
		foreach (SerializablePlayer player in runLobby.Run.Players)
		{
			if (player.NetId != _lobby.NetService.NetId || displayLocalPlayer)
			{
				NRemoteLobbyPlayer nRemoteLobbyPlayer = NRemoteLobbyPlayer.Create(runLobby, player.NetId, runLobby.NetService.Platform, runLobby.NetService.Type == NetGameType.Singleplayer);
				_container.AddChildSafely(nRemoteLobbyPlayer);
				_nodes.Add(nRemoteLobbyPlayer);
			}
		}
	}

	public void OnPlayerConnected(ulong playerId)
	{
		OnPlayerChanged(playerId);
	}

	public void OnPlayerDisconnected(ulong playerId)
	{
		OnPlayerChanged(playerId);
	}

	public void OnPlayerChanged(ulong playerId)
	{
		if (_lobby != null && playerId != _lobby.NetService.NetId)
		{
			int num = _nodes.FindIndex((NRemoteLobbyPlayer p) => p.PlayerId == playerId);
			if (num >= 0)
			{
				NRemoteLobbyPlayer nRemoteLobbyPlayer = _nodes[num];
				nRemoteLobbyPlayer.OnPlayerChanged(_lobby, playerId);
			}
		}
	}

	public void Cleanup()
	{
		foreach (NRemoteLobbyPlayer node in _nodes)
		{
			node.QueueFreeSafely();
		}
		_nodes.Clear();
	}
}
