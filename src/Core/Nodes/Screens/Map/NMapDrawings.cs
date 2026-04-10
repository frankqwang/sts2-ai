using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Multiplayer.Game.PeerInput;
using MegaCrit.Sts2.Core.Multiplayer.Messages.Game.Flavor;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves.MapDrawing;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Map;

public partial class NMapDrawings : Control
{
	private class DrawingState
	{
		public DrawingMode? overrideDrawingMode;

		public DrawingMode drawingMode;

		public ulong playerId;

		public Line2D? currentlyDrawingLine;

		public required SubViewport drawViewport;

		public bool IsDrawing => currentlyDrawingLine != null;

		public DrawingMode CurrentDrawingMode => overrideDrawingMode ?? drawingMode;
	}

	private const int _minUpdateMsec = 50;

	private static readonly string _lineDrawScenePath = SceneHelper.GetScenePath("screens/map/map_line_draw");

	private static readonly string _lineEraseScenePath = SceneHelper.GetScenePath("screens/map/map_line_erase");

	private static readonly string _playerDrawingPath = SceneHelper.GetScenePath("screens/map/map_drawing");

	public const string drawingCursorPath = "res://images/packed/common_ui/cursor_quill.png";

	public const string drawingCursorTiltedPath = "res://images/packed/common_ui/cursor_quill_tilted.png";

	public static readonly Vector2 drawingCursorHotspot = new Vector2(2f, 56f);

	public const string erasingCursorPath = "res://images/packed/common_ui/cursor_eraser.png";

	public const string erasingCursorTiltedPath = "res://images/packed/common_ui/cursor_eraser_tilted.png";

	public static readonly Vector2 erasingCursorHotspot = new Vector2(24f, 58f);

	private const float _minimumPointDistance = 2f;

	private INetGameService _netService;

	private IPlayerCollection _playerCollection;

	private PeerInputSynchronizer _inputSynchronizer;

	private PackedScene _lineDrawScene;

	private PackedScene _lineEraseScene;

	private NCursorManager _cursorManager;

	private Material _eraserMaterial;

	private Vector2 _defaultSize;

	private readonly List<DrawingState> _drawingStates = new List<DrawingState>();

	private MapDrawingMessage? _queuedMessage;

	private ulong _lastMessageMsec;

	private Task? _sendMessageTask;

	private static IEnumerable<string> SelfAssetPaths => new global::_003C_003Ez__ReadOnlyArray<string>(new string[7] { _lineDrawScenePath, _lineEraseScenePath, _playerDrawingPath, "res://images/packed/common_ui/cursor_quill.png", "res://images/packed/common_ui/cursor_quill_tilted.png", "res://images/packed/common_ui/cursor_eraser.png", "res://images/packed/common_ui/cursor_eraser_tilted.png" });

	public static IEnumerable<string> AssetPaths => SelfAssetPaths.Concat(NMapDrawButton.AssetPaths);

	public override void _Ready()
	{
		_lineDrawScene = PreloadManager.Cache.GetScene(_lineDrawScenePath);
		_lineEraseScene = PreloadManager.Cache.GetScene(_lineEraseScenePath);
		_cursorManager = NGame.Instance.CursorManager;
		Line2D line2D = _lineEraseScene.Instantiate<Line2D>(PackedScene.GenEditState.Disabled);
		_eraserMaterial = line2D.Material;
		line2D.QueueFreeSafely();
		_defaultSize = base.Size;
	}

	public void Initialize(INetGameService netService, IPlayerCollection playerCollection, PeerInputSynchronizer inputSynchronizer)
	{
		_netService = netService;
		_playerCollection = playerCollection;
		_inputSynchronizer = inputSynchronizer;
		_netService.RegisterMessageHandler<MapDrawingMessage>(HandleDrawingMessage);
		_netService.RegisterMessageHandler<ClearMapDrawingsMessage>(HandleClearMapDrawingsMessage);
		_netService.RegisterMessageHandler<MapDrawingModeChangedMessage>(HandleMapDrawingModeChangedMessage);
		inputSynchronizer.ScreenChanged += OnPlayerScreenChanged;
	}

	public override void _ExitTree()
	{
		_netService.UnregisterMessageHandler<MapDrawingMessage>(HandleDrawingMessage);
		_netService.UnregisterMessageHandler<ClearMapDrawingsMessage>(HandleClearMapDrawingsMessage);
		_netService.UnregisterMessageHandler<MapDrawingModeChangedMessage>(HandleMapDrawingModeChangedMessage);
		_inputSynchronizer.ScreenChanged -= OnPlayerScreenChanged;
	}

	public void BeginLineLocal(Vector2 position, DrawingMode? overrideDrawingMode)
	{
		BeginLine(GetDrawingStateForPlayer(_netService.NetId), position, overrideDrawingMode);
		NetMapDrawingEvent ev = new NetMapDrawingEvent
		{
			type = MapDrawingEventType.BeginLine,
			position = ToNetPosition(position),
			overrideDrawingMode = overrideDrawingMode
		};
		QueueOrSendEvent(ev);
	}

	public void UpdateCurrentLinePositionLocal(Vector2 position)
	{
		DrawingState drawingStateForPlayer = GetDrawingStateForPlayer(_netService.NetId);
		UpdateCurrentLinePosition(drawingStateForPlayer, position);
		NetMapDrawingEvent ev = new NetMapDrawingEvent
		{
			type = MapDrawingEventType.ContinueLine,
			position = ToNetPosition(position),
			overrideDrawingMode = drawingStateForPlayer.overrideDrawingMode
		};
		QueueOrSendEvent(ev);
	}

	public void StopLineLocal()
	{
		StopDrawingLine(GetDrawingStateForPlayer(_netService.NetId));
		NetMapDrawingEvent ev = new NetMapDrawingEvent
		{
			type = MapDrawingEventType.EndLine
		};
		QueueOrSendEvent(ev);
	}

	public void SetDrawingModeLocal(DrawingMode drawingMode)
	{
		SetDrawingMode(GetDrawingStateForPlayer(_netService.NetId), drawingMode);
		MapDrawingModeChangedMessage message = new MapDrawingModeChangedMessage
		{
			drawingMode = drawingMode
		};
		_netService.SendMessage(message);
		UpdateLocalCursor();
	}

	public void ClearDrawnLinesLocal()
	{
		ClearAllLinesForPlayer(GetDrawingStateForPlayer(_netService.NetId));
		UpdateLocalCursor();
		_netService.SendMessage(default(ClearMapDrawingsMessage));
	}

	public bool IsDrawing(ulong playerId)
	{
		return GetDrawingStateForPlayer(playerId).IsDrawing;
	}

	public bool IsLocalDrawing()
	{
		return GetDrawingStateForPlayer(_netService.NetId).IsDrawing;
	}

	public DrawingMode GetDrawingMode(ulong playerId)
	{
		return GetDrawingStateForPlayer(playerId).CurrentDrawingMode;
	}

	public DrawingMode GetLocalDrawingMode(bool useOverride = true)
	{
		if (!useOverride)
		{
			return GetDrawingStateForPlayer(_netService.NetId).drawingMode;
		}
		return GetDrawingStateForPlayer(_netService.NetId).CurrentDrawingMode;
	}

	private void QueueOrSendEvent(NetMapDrawingEvent ev)
	{
		if (_queuedMessage == null)
		{
			_queuedMessage = new MapDrawingMessage();
		}
		if (!_queuedMessage.TryAddEvent(ev))
		{
			_queuedMessage.drawingMode = GetDrawingStateForPlayer(_netService.NetId).drawingMode;
			_netService.SendMessage(_queuedMessage);
			_queuedMessage = new MapDrawingMessage();
			if (!_queuedMessage.TryAddEvent(ev))
			{
				throw new InvalidOperationException();
			}
		}
		TrySendSyncMessage();
		UpdateLocalCursor();
	}

	private Vector2 ToNetPosition(Vector2 pos)
	{
		pos.X -= base.Size.X * 0.5f;
		pos /= new Vector2(960f, base.Size.Y);
		return pos;
	}

	private Vector2 FromNetPosition(Vector2 pos)
	{
		pos *= new Vector2(960f, base.Size.Y);
		pos.X += base.Size.X * 0.5f;
		return pos;
	}

	private void HandleDrawingMessage(MapDrawingMessage message, ulong senderId)
	{
		DrawingState drawingStateForPlayer = GetDrawingStateForPlayer(senderId);
		foreach (NetMapDrawingEvent @event in message.Events)
		{
			if (@event.type == MapDrawingEventType.BeginLine)
			{
				if (GetDrawingMode(senderId) != DrawingMode.None)
				{
					StopDrawingLine(drawingStateForPlayer);
				}
				BeginLine(drawingStateForPlayer, FromNetPosition(@event.position), @event.overrideDrawingMode);
			}
			else if (@event.type == MapDrawingEventType.ContinueLine)
			{
				if (!drawingStateForPlayer.IsDrawing)
				{
					if (message.drawingMode.HasValue && drawingStateForPlayer.drawingMode != message.drawingMode)
					{
						SetDrawingMode(drawingStateForPlayer, message.drawingMode.Value);
					}
					BeginLine(drawingStateForPlayer, FromNetPosition(@event.position), @event.overrideDrawingMode);
				}
				UpdateCurrentLinePosition(drawingStateForPlayer, FromNetPosition(@event.position));
			}
			else
			{
				StopDrawingLine(drawingStateForPlayer);
			}
		}
	}

	private void HandleClearMapDrawingsMessage(ClearMapDrawingsMessage message, ulong senderId)
	{
		ClearAllLinesForPlayer(GetDrawingStateForPlayer(senderId));
	}

	private void HandleMapDrawingModeChangedMessage(MapDrawingModeChangedMessage message, ulong senderId)
	{
		SetDrawingMode(GetDrawingStateForPlayer(senderId), message.drawingMode);
	}

	private void BeginLine(DrawingState state, Vector2 position, DrawingMode? overrideDrawingMode)
	{
		Player player = _playerCollection.GetPlayer(state.playerId);
		DrawingMode drawingMode = overrideDrawingMode ?? state.drawingMode;
		if (drawingMode == DrawingMode.None)
		{
			throw new InvalidOperationException($"Player {state.playerId} is not currently in a drawing mode and no override was passed!");
		}
		state.overrideDrawingMode = overrideDrawingMode;
		state.currentlyDrawingLine = CreateLineForPlayer(player, drawingMode == DrawingMode.Erasing);
		state.currentlyDrawingLine.AddPoint(position * 0.5f);
		state.currentlyDrawingLine.AddPoint(position * 0.5f + new Vector2(0f, 0.5f));
		state.drawViewport.AddChildSafely(state.currentlyDrawingLine);
		NGame.Instance.RemoteCursorContainer.DrawingCursorStateChanged(state.playerId);
	}

	private Line2D CreateLineForPlayer(Player player, bool isErasing)
	{
		PackedScene packedScene = (isErasing ? _lineEraseScene : _lineDrawScene);
		Line2D line2D = packedScene.Instantiate<Line2D>(PackedScene.GenEditState.Disabled);
		line2D.DefaultColor = player.Character.MapDrawingColor;
		line2D.ClearPoints();
		line2D.Position = Vector2.Zero;
		return line2D;
	}

	private void StopDrawingLine(DrawingState state)
	{
		state.overrideDrawingMode = null;
		state.currentlyDrawingLine = null;
		NGame.Instance.RemoteCursorContainer.DrawingCursorStateChanged(state.playerId);
	}

	private void SetDrawingMode(DrawingState state, DrawingMode drawingMode)
	{
		if (state.drawingMode != drawingMode)
		{
			state.drawingMode = drawingMode;
			NGame.Instance.RemoteCursorContainer.DrawingCursorStateChanged(state.playerId);
		}
	}

	private void UpdateCurrentLinePosition(DrawingState state, Vector2 position)
	{
		if (state.currentlyDrawingLine == null)
		{
			throw new InvalidOperationException($"Tried to update current line position for player {state.playerId}, but they are not currently drawing a line!");
		}
		Vector2 vector = state.currentlyDrawingLine.Points[^1];
		if (!(vector.DistanceSquaredTo(position) < 4f))
		{
			state.currentlyDrawingLine.AddPoint(position * 0.5f);
		}
	}

	private DrawingState GetDrawingStateForPlayer(ulong playerId)
	{
		DrawingState drawingState = _drawingStates.FirstOrDefault((DrawingState s) => s.playerId == playerId);
		if (drawingState == null)
		{
			Control control = PreloadManager.Cache.GetScene(_playerDrawingPath).Instantiate<Control>(PackedScene.GenEditState.Disabled);
			this.AddChildSafely(control);
			drawingState = new DrawingState
			{
				playerId = playerId,
				drawViewport = control.GetNode<SubViewport>("DrawViewport")
			};
			TaskHelper.RunSafely(SetVisibleLater(control));
			_drawingStates.Add(drawingState);
		}
		return drawingState;
	}

	private async Task SetVisibleLater(Control mapDrawingScene)
	{
		TextureRect drawingTexture = mapDrawingScene.GetNode<TextureRect>("DrawViewportTextureRect");
		SubViewport drawViewport = mapDrawingScene.GetNode<SubViewport>("DrawViewport");
		drawViewport.RenderTargetUpdateMode = SubViewport.UpdateMode.Always;
		drawingTexture.Visible = false;
		await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		drawingTexture.Visible = true;
		drawViewport.RenderTargetUpdateMode = SubViewport.UpdateMode.WhenVisible;
	}

	public void ClearAllLines()
	{
		foreach (DrawingState drawingState in _drawingStates)
		{
			foreach (Line2D item in drawingState.drawViewport.GetChildren().OfType<Line2D>())
			{
				item.QueueFreeSafely();
			}
		}
	}

	public SerializableMapDrawings GetSerializableMapDrawings()
	{
		SerializableMapDrawings serializableMapDrawings = new SerializableMapDrawings();
		foreach (DrawingState drawingState in _drawingStates)
		{
			SerializablePlayerMapDrawings serializablePlayerMapDrawings = new SerializablePlayerMapDrawings
			{
				playerId = drawingState.playerId
			};
			serializableMapDrawings.drawings.Add(serializablePlayerMapDrawings);
			foreach (Line2D item in drawingState.drawViewport.GetChildren().OfType<Line2D>())
			{
				SerializableMapDrawingLine serializableMapDrawingLine = new SerializableMapDrawingLine
				{
					mapPoints = new List<Vector2>()
				};
				serializableMapDrawingLine.isEraser = item.Material == _eraserMaterial;
				serializablePlayerMapDrawings.lines.Add(serializableMapDrawingLine);
				Vector2[] points = item.Points;
				foreach (Vector2 pos in points)
				{
					serializableMapDrawingLine.mapPoints.Add(ToNetPosition(pos));
				}
			}
		}
		return serializableMapDrawings;
	}

	public void LoadDrawings(SerializableMapDrawings drawings)
	{
		foreach (SerializablePlayerMapDrawings drawing in drawings.drawings)
		{
			Player player = _playerCollection.GetPlayer(drawing.playerId);
			if (player == null)
			{
				Log.Warn($"Player {drawing.playerId} has map drawings, but doesn't exist in the run!");
				continue;
			}
			DrawingState drawingStateForPlayer = GetDrawingStateForPlayer(drawing.playerId);
			foreach (SerializableMapDrawingLine line in drawing.lines)
			{
				Line2D line2D = CreateLineForPlayer(player, line.isEraser);
				drawingStateForPlayer.drawViewport.AddChildSafely(line2D);
				foreach (Vector2 mapPoint in line.mapPoints)
				{
					line2D.AddPoint(FromNetPosition(mapPoint));
				}
			}
		}
	}

	private void ClearAllLinesForPlayer(DrawingState state)
	{
		foreach (Line2D item in state.drawViewport.GetChildren().OfType<Line2D>())
		{
			item.QueueFreeSafely();
		}
		SetDrawingMode(state, DrawingMode.None);
	}

	private void OnPlayerScreenChanged(ulong playerId, NetScreenType oldScreenType)
	{
		if (playerId == _netService.NetId)
		{
			return;
		}
		NetScreenType screenType = _inputSynchronizer.GetScreenType(playerId);
		if (oldScreenType == NetScreenType.Map && screenType != NetScreenType.Map)
		{
			DrawingState drawingStateForPlayer = GetDrawingStateForPlayer(playerId);
			if (drawingStateForPlayer.IsDrawing)
			{
				StopDrawingLine(drawingStateForPlayer);
			}
			if (drawingStateForPlayer.drawingMode != DrawingMode.None)
			{
				SetDrawingMode(drawingStateForPlayer, DrawingMode.None);
			}
		}
	}

	private void TrySendSyncMessage()
	{
		if (_sendMessageTask == null && _netService.IsConnected)
		{
			int num = (int)(_lastMessageMsec + 50 - Time.GetTicksMsec());
			if (num <= 0)
			{
				_sendMessageTask = TaskHelper.RunSafely(SendSyncMessageAfterSmallDelay());
			}
			else
			{
				_sendMessageTask = TaskHelper.RunSafely(QueueSyncMessage(num));
			}
		}
	}

	private async Task QueueSyncMessage(int delayMsec)
	{
		await Task.Delay(delayMsec);
		SendSyncMessage();
	}

	private async Task SendSyncMessageAfterSmallDelay()
	{
		await Task.Yield();
		SendSyncMessage();
	}

	private void SendSyncMessage()
	{
		if (_netService.IsConnected)
		{
			_queuedMessage.drawingMode = GetDrawingStateForPlayer(_netService.NetId).drawingMode;
			_netService.SendMessage(_queuedMessage);
			_lastMessageMsec = Time.GetTicksMsec();
			_queuedMessage = null;
			_sendMessageTask = null;
		}
	}

	private void UpdateLocalCursor()
	{
		DrawingState drawingStateForPlayer = GetDrawingStateForPlayer(_netService.NetId);
		if (drawingStateForPlayer.CurrentDrawingMode == DrawingMode.Drawing)
		{
			Image asset = PreloadManager.Cache.GetAsset<Image>("res://images/packed/common_ui/cursor_quill.png");
			Image asset2 = PreloadManager.Cache.GetAsset<Image>("res://images/packed/common_ui/cursor_quill_tilted.png");
			_cursorManager.OverrideCursor(asset2, asset, drawingCursorHotspot);
		}
		else if (drawingStateForPlayer.CurrentDrawingMode == DrawingMode.Erasing)
		{
			Image asset3 = PreloadManager.Cache.GetAsset<Image>("res://images/packed/common_ui/cursor_eraser.png");
			Image asset4 = PreloadManager.Cache.GetAsset<Image>("res://images/packed/common_ui/cursor_eraser_tilted.png");
			_cursorManager.OverrideCursor(asset4, asset3, erasingCursorHotspot);
		}
		else
		{
			_cursorManager.StopOverridingCursor();
		}
	}

	public void RepositionBasedOnBackground(Control mapBg)
	{
		base.Position = new Vector2(mapBg.Position.X + (mapBg.Size.X - base.Size.X) * 0.5f, mapBg.Position.Y);
	}
}
