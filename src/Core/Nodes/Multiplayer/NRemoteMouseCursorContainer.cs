using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.Debug;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Multiplayer.Game.PeerInput;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Vfx;

namespace MegaCrit.Sts2.Core.Nodes.Multiplayer;

public partial class NRemoteMouseCursorContainer : Control
{
	private static bool _isDebugUiVisible = true;

	private PeerInputSynchronizer? _synchronizer;

	private readonly List<NRemoteMouseCursor> _cursors = new List<NRemoteMouseCursor>();

	public void Initialize(PeerInputSynchronizer synchronizer, IEnumerable<ulong> connectedPlayerIds)
	{
		if (_synchronizer != null)
		{
			Deinitialize();
		}
		_synchronizer = synchronizer;
		_synchronizer.StateAdded += OnInputStateAdded;
		_synchronizer.StateChanged += OnInputStateChanged;
		_synchronizer.StateRemoved += OnInputStateRemoved;
		_synchronizer.NetService.Disconnected += NetServiceDisconnected;
	}

	private void NetServiceDisconnected(NetErrorInfo _)
	{
		Deinitialize();
	}

	public void Deinitialize()
	{
		if (_synchronizer != null)
		{
			_synchronizer.StateAdded -= OnInputStateAdded;
			_synchronizer.StateChanged -= OnInputStateChanged;
			_synchronizer.StateRemoved -= OnInputStateRemoved;
			_synchronizer.NetService.Disconnected -= NetServiceDisconnected;
			_synchronizer.Dispose();
			_synchronizer = null;
		}
		foreach (NRemoteMouseCursor cursor in _cursors)
		{
			cursor.QueueFreeSafely();
		}
		_cursors.Clear();
	}

	public override void _Ready()
	{
		GetViewport().Connect(Viewport.SignalName.GuiFocusChanged, Callable.From<Control>(OnGuiFocusChanged));
	}

	public override void _ExitTree()
	{
		Deinitialize();
	}

	public void ForceUpdateAllCursors()
	{
		foreach (NRemoteMouseCursor cursor in _cursors)
		{
			OnInputStateChanged(cursor.PlayerId);
		}
	}

	public Vector2 GetCursorPosition(ulong playerId)
	{
		return GetCursor(playerId).Position;
	}

	private void OnInputStateAdded(ulong playerId)
	{
		AddCursor(playerId);
	}

	private void OnInputStateRemoved(ulong playerId)
	{
		RemoveCursor(playerId);
	}

	private void AddCursor(ulong playerId)
	{
		if (playerId != _synchronizer?.NetService.NetId)
		{
			if (_cursors.Any((NRemoteMouseCursor c) => c.PlayerId == playerId))
			{
				Log.Error($"Tried to add cursor for player {playerId} twice!");
			}
			else
			{
				NRemoteMouseCursor nRemoteMouseCursor = NRemoteMouseCursor.Create(playerId);
				_cursors.Add(nRemoteMouseCursor);
				this.AddChildSafely(nRemoteMouseCursor);
			}
		}
	}

	private void OnInputStateChanged(ulong playerId)
	{
		if (playerId == _synchronizer?.NetService.NetId)
		{
			UpdateCursorVisibility();
			return;
		}
		Vector2 controlSpaceFocusPosition = _synchronizer.GetControlSpaceFocusPosition(playerId, this);
		NRemoteMouseCursor cursor = GetCursor(playerId);
		cursor.SetNextPosition(controlSpaceFocusPosition);
		cursor.UpdateImage(_synchronizer.GetMouseDown(playerId), GetDrawingMode(playerId));
		UpdateCursorVisibility();
	}

	public void DrawingCursorStateChanged(ulong playerId)
	{
		GetCursor(playerId)?.UpdateImage(_synchronizer.GetMouseDown(playerId), GetDrawingMode(playerId));
	}

	private static DrawingMode GetDrawingMode(ulong playerId)
	{
		if (NRun.Instance != null)
		{
			return NRun.Instance.GlobalUi.MapScreen.Drawings.GetDrawingMode(playerId);
		}
		return DrawingMode.None;
	}

	private NRemoteMouseCursor? GetCursor(ulong playerId)
	{
		return _cursors.FirstOrDefault((NRemoteMouseCursor c) => c.PlayerId == playerId);
	}

	private void RemoveCursor(ulong playerId)
	{
		NRemoteMouseCursor cursor = GetCursor(playerId);
		if (cursor != null)
		{
			cursor.QueueFreeSafely();
			_cursors.Remove(cursor);
		}
	}

	private void UpdateCursorVisibility()
	{
		NetScreenType screenType = _synchronizer.GetScreenType(_synchronizer.NetService.NetId);
		foreach (NRemoteMouseCursor cursor in _cursors)
		{
			NetScreenType screenType2 = _synchronizer.GetScreenType(cursor.PlayerId);
			bool flag = screenType == screenType2;
			bool flag2 = (((uint)(screenType2 - 5) <= 3u || screenType2 == NetScreenType.RemotePlayerExpandedState) ? true : false);
			bool flag3 = flag2;
			bool flag4 = screenType2 == NetScreenType.SharedRelicPicking;
			cursor.Visible = flag && !flag3 && !flag4;
		}
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (_synchronizer == null)
		{
			return;
		}
		if (inputEvent.IsActionReleased(DebugHotkey.hideMpCursors))
		{
			_isDebugUiVisible = !_isDebugUiVisible;
			ApplyDebugUiVisibility();
			NGame.Instance.AddChildSafely(NFullscreenTextVfx.Create(_isDebugUiVisible ? "Show MP Cursors" : "Hide MP Cursors"));
		}
		if (inputEvent is InputEventMouseMotion inputEventMouseMotion)
		{
			_synchronizer.SyncLocalIsUsingController(isUsingController: false);
			if (!NGame.Instance.ReactionWheel.Visible)
			{
				_synchronizer.SyncLocalMousePos(inputEventMouseMotion.Position, this);
			}
		}
		else if (inputEvent is InputEventMouseButton inputEventMouseButton)
		{
			_synchronizer.SyncLocalIsUsingController(isUsingController: false);
			if (inputEventMouseButton.ButtonIndex == MouseButton.Left)
			{
				_synchronizer.SyncLocalMouseDown(inputEventMouseButton.Pressed);
			}
		}
	}

	private void OnGuiFocusChanged(Control focused)
	{
		if (_synchronizer != null)
		{
			NControllerManager? instance = NControllerManager.Instance;
			if (instance != null && instance.IsUsingController)
			{
				_synchronizer.SyncLocalIsUsingController(isUsingController: true);
				_synchronizer.SyncLocalControllerFocus(focused.GlobalPosition + focused.Size * 0.5f, this);
			}
		}
	}

	private void ApplyDebugUiVisibility()
	{
		base.Visible = _isDebugUiVisible;
	}
}
