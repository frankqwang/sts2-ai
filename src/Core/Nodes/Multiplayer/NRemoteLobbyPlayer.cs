using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Game.Lobby;
using MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;
using MegaCrit.Sts2.Core.Platform;
using MegaCrit.Sts2.Core.Saves.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Multiplayer;

public partial class NRemoteLobbyPlayer : Control
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("ui/remote_lobby_player");

	private TextureRect _characterIcon;

	private Control _readyIndicator;

	private Control _disconnectedIndicator;

	private MegaLabel _nameplateLabel;

	private MegaLabel _characterLabel;

	private PlatformType _platform;

	private bool _isSingleplayer;

	private ScreenPunchInstance? _shake;

	private Vector2? _originalPosition;

	private ulong _playerId;

	private CharacterModel _character;

	private bool _isReady;

	private bool _isConnected;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public ulong PlayerId => _playerId;

	public static NRemoteLobbyPlayer Create(LobbyPlayer player, PlatformType platform, bool isSingleplayer)
	{
		NRemoteLobbyPlayer nRemoteLobbyPlayer = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NRemoteLobbyPlayer>(PackedScene.GenEditState.Disabled);
		nRemoteLobbyPlayer._playerId = player.id;
		nRemoteLobbyPlayer._platform = platform;
		nRemoteLobbyPlayer._isSingleplayer = isSingleplayer;
		nRemoteLobbyPlayer._character = player.character;
		nRemoteLobbyPlayer._isReady = player.isReady;
		nRemoteLobbyPlayer._isConnected = true;
		return nRemoteLobbyPlayer;
	}

	public static NRemoteLobbyPlayer Create(LoadRunLobby runLobby, ulong playerId, PlatformType platform, bool isSingleplayer)
	{
		NRemoteLobbyPlayer nRemoteLobbyPlayer = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NRemoteLobbyPlayer>(PackedScene.GenEditState.Disabled);
		nRemoteLobbyPlayer._playerId = playerId;
		nRemoteLobbyPlayer._isSingleplayer = isSingleplayer;
		nRemoteLobbyPlayer._platform = runLobby.NetService.Platform;
		nRemoteLobbyPlayer._character = ModelDb.GetById<CharacterModel>(runLobby.Run.Players.First((SerializablePlayer p) => p.NetId == playerId).CharacterId);
		nRemoteLobbyPlayer._isReady = runLobby.IsPlayerReady(playerId);
		nRemoteLobbyPlayer._isConnected = runLobby.ConnectedPlayerIds.Contains(playerId);
		return nRemoteLobbyPlayer;
	}

	public override void _Ready()
	{
		_nameplateLabel = GetNode<MegaLabel>("%NameplateLabel");
		_characterLabel = GetNode<MegaLabel>("%CharacterLabel");
		_characterIcon = GetNode<TextureRect>("%CharacterIcon");
		_readyIndicator = GetNode<TextureRect>("%ReadyIndicator");
		_disconnectedIndicator = GetNode<TextureRect>("%DisconnectedIndicator");
		if (!_isSingleplayer)
		{
			_nameplateLabel.SetTextAutoSize(PlatformUtil.GetPlayerName(_platform, _playerId));
		}
		else
		{
			_characterLabel.SetTextAutoSize(string.Empty);
		}
		RefreshVisuals();
	}

	public void OnPlayerChanged(LobbyPlayer lobbyPlayer)
	{
		_playerId = lobbyPlayer.id;
		SetCharacter(lobbyPlayer.character);
		_isReady = lobbyPlayer.isReady;
		_isConnected = true;
		RefreshVisuals();
	}

	public void OnPlayerChanged(LoadRunLobby runLobby, ulong playerId)
	{
		SerializablePlayer serializablePlayer = runLobby.Run.Players.First((SerializablePlayer p) => p.NetId == playerId);
		SetCharacter(ModelDb.GetById<CharacterModel>(serializablePlayer.CharacterId));
		_isReady = runLobby.IsPlayerReady(playerId);
		_isConnected = runLobby.ConnectedPlayerIds.Contains(playerId);
		RefreshVisuals();
	}

	private void RefreshVisuals()
	{
		if (_isSingleplayer)
		{
			_nameplateLabel.SetTextAutoSize(_character.Title.GetFormattedText());
		}
		else
		{
			_characterLabel.SetTextAutoSize(_character.Title.GetFormattedText());
		}
		_characterIcon.Texture = _character.IconTexture;
		_readyIndicator.Visible = _isReady;
		_disconnectedIndicator.Visible = !_isConnected;
	}

	private void SetCharacter(CharacterModel character)
	{
		if (_character != character)
		{
			_shake?.Cancel();
			CancelShake();
			_originalPosition = base.Position;
			_shake = new ScreenPunchInstance(3f, 0.4000000059604645, 90f);
			_character = character;
		}
	}

	public void CancelShake()
	{
		_shake = null;
		if (_originalPosition.HasValue)
		{
			base.Position = _originalPosition.Value;
			_originalPosition = null;
		}
	}

	public override void _Process(double delta)
	{
		ScreenPunchInstance shake = _shake;
		if (shake != null && !shake.IsDone)
		{
			Vector2 vector = _shake?.Update(delta) ?? Vector2.Zero;
			base.Position = _originalPosition.Value + vector;
		}
	}
}
