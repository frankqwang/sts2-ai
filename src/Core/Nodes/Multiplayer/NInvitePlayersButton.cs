using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Multiplayer.Game.Lobby;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Platform;

namespace MegaCrit.Sts2.Core.Nodes.Multiplayer;

public partial class NInvitePlayersButton : NButton
{
	private static readonly StringName _v = new StringName("v");

	private static readonly StringName _s = new StringName("s");

	private ShaderMaterial _shaderMaterial;

	private Control _container;

	private StartRunLobby? _startRunLobby;

	public override void _Ready()
	{
		ConnectSignals();
		_container = GetParent<Control>();
		Control node = GetNode<Control>("Background");
		MegaRichTextLabel node2 = GetNode<MegaRichTextLabel>("Label");
		_shaderMaterial = (ShaderMaterial)node.Material;
		node2.SetTextAutoSize(new LocString("main_menu_ui", "INVITE").GetFormattedText());
		UpdateVisibility();
	}

	public void Initialize(StartRunLobby lobby)
	{
		_startRunLobby = lobby;
		_startRunLobby.PlayerConnected += OnPlayerConnected;
		_startRunLobby.PlayerDisconnected += OnPlayerDisconnected;
		UpdateVisibility();
	}

	public override void _ExitTree()
	{
		if (_startRunLobby != null)
		{
			_startRunLobby.PlayerConnected -= OnPlayerConnected;
			_startRunLobby.PlayerDisconnected -= OnPlayerDisconnected;
		}
	}

	private void OnPlayerConnected(LobbyPlayer player)
	{
		UpdateVisibility();
	}

	private void OnPlayerDisconnected(LobbyPlayer player)
	{
		UpdateVisibility();
	}

	private void UpdateVisibility()
	{
		_container.Visible = _startRunLobby != null && PlatformUtil.SupportsInviteDialog(_startRunLobby.NetService.Platform) && _startRunLobby.Players.Count < _startRunLobby.MaxPlayers;
	}

	protected override void OnRelease()
	{
		if (_startRunLobby != null)
		{
			PlatformUtil.OpenInviteDialog(_startRunLobby.NetService);
		}
	}

	protected override void OnFocus()
	{
		_shaderMaterial.SetShaderParameter(_s, 1.1f);
		_shaderMaterial.SetShaderParameter(_v, 1.1f);
	}

	protected override void OnUnfocus()
	{
		_shaderMaterial.SetShaderParameter(_s, 1f);
		_shaderMaterial.SetShaderParameter(_v, 1f);
	}
}
