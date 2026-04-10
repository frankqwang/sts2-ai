using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Multiplayer.Game.Lobby;

namespace MegaCrit.Sts2.Core.Nodes.Multiplayer;

public partial class NRemoteLobbyPlayerContainer : Control
{
	private readonly List<NRemoteLobbyPlayer> _nodes = new List<NRemoteLobbyPlayer>();

	private StartRunLobby? _lobby;

	private NInvitePlayersButton _inviteButton;

	private MegaLabel _soloLabel;

	private Container _container;

	private bool _displayLocalPlayer;

	public override void _Ready()
	{
		_soloLabel = GetNode<MegaLabel>("%SoloLabel");
		_container = GetNode<Container>("Container");
		_inviteButton = GetNode<NInvitePlayersButton>("%InviteButton");
		_soloLabel.SetTextAutoSize(new LocString("main_menu_ui", "MULTIPLAYER_CHAR_SELECT.SOLO").GetFormattedText());
	}

	public void Initialize(StartRunLobby lobby, bool displayLocalPlayer)
	{
		foreach (NRemoteLobbyPlayer node in _nodes)
		{
			node.QueueFreeSafely();
		}
		_nodes.Clear();
		if (!lobby.NetService.Type.IsMultiplayer())
		{
			return;
		}
		_displayLocalPlayer = displayLocalPlayer;
		_inviteButton.Initialize(lobby);
		_lobby = lobby;
		foreach (LobbyPlayer player in _lobby.Players)
		{
			OnPlayerConnected(player);
		}
		RefreshSoloLabelVisibility();
	}

	public void OnPlayerConnected(LobbyPlayer player)
	{
		StartRunLobby lobby = _lobby;
		if (lobby != null && (player.id != lobby.LocalPlayer.id || _displayLocalPlayer))
		{
			NRemoteLobbyPlayer nRemoteLobbyPlayer = NRemoteLobbyPlayer.Create(player, lobby.NetService.Platform, lobby.NetService.Type == NetGameType.Singleplayer);
			_container.AddChildSafely(nRemoteLobbyPlayer);
			_container.MoveChild(_inviteButton.GetParent(), _container.GetChildCount() - 1);
			_nodes.Add(nRemoteLobbyPlayer);
			RefreshSoloLabelVisibility();
		}
	}

	public void OnPlayerDisconnected(LobbyPlayer player)
	{
		if (_lobby == null)
		{
			return;
		}
		int num = _nodes.FindIndex((NRemoteLobbyPlayer p) => p.PlayerId == player.id);
		if (num >= 0)
		{
			_container.RemoveChildSafely(_nodes[num]);
			_nodes.RemoveAt(num);
			foreach (NRemoteLobbyPlayer node in _nodes)
			{
				node.CancelShake();
			}
		}
		RefreshSoloLabelVisibility();
	}

	public void OnPlayerChanged(LobbyPlayer player)
	{
		StartRunLobby lobby = _lobby;
		if (lobby != null && (player.id != lobby.LocalPlayer.id || _displayLocalPlayer))
		{
			_nodes.FirstOrDefault((NRemoteLobbyPlayer p) => p.PlayerId == player.id)?.OnPlayerChanged(player);
		}
	}

	private void RefreshSoloLabelVisibility()
	{
		StartRunLobby lobby = _lobby;
		_soloLabel.Visible = (lobby == null || lobby.NetService.Type != NetGameType.Singleplayer) && lobby != null && lobby.Players.Count == 1;
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
