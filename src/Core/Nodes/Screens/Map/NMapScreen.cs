using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Debug;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models.Modifiers;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Multiplayer.Game.PeerInput;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Ftue;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.Capstones;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Nodes.Vfx;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Map;

public partial class NMapScreen : Control, IScreenContext, INetCursorPositionTranslator
{
	[Signal]
	public delegate void OpenedEventHandler();

	[Signal]
	public delegate void ClosedEventHandler();

	private ActMap _map = NullActMap.Instance;

	private Control _mapContainer;

	private Control _pathsContainer;

	private Control _points;

	private NBossMapPoint? _bossPointNode;

	private NBossMapPoint? _secondBossPointNode;

	private NMapPoint? _startingPointNode;

	private NMapBg _mapBgContainer;

	private NMapMarker _marker;

	private NBackButton _backButton;

	private TextureRect _drawingToolsHotkeyIcon;

	private Control _drawingTools;

	private NMapDrawButton _mapDrawingButton;

	private NMapEraseButton _mapErasingButton;

	private NMapClearButton _mapClearButton;

	private Control _mapLegend;

	private Control _legendItems;

	private TextureRect _legendHotkeyIcon;

	private Control _backstop;

	private Tween? _tween;

	private Vector2 _startDragPos;

	private Vector2 _targetDragPos;

	private bool _isDragging;

	private bool _hasPlayedAnimation;

	private readonly Dictionary<MapCoord, NMapPoint> _mapPointDictionary = new Dictionary<MapCoord, NMapPoint>();

	private readonly Dictionary<(MapCoord, MapCoord), IReadOnlyList<TextureRect>> _paths = new Dictionary<(MapCoord, MapCoord), IReadOnlyList<TextureRect>>();

	private float _controllerScrollAmount = 400f;

	private const float _scrollLimitTop = 1800f;

	private const float _scrollLimitBottom = -600f;

	private const float _totalHeight = 2325f;

	private const float _totalWidth = 1050f;

	private float _distX;

	private float _distY;

	private const float _pointJitterX = 21f;

	private const float _pointJitterY = 25f;

	private const float _tickDist = 22f;

	private const float _pathPosJitter = 3f;

	private const float _pathAngleJitter = 0.1f;

	private static readonly Vector2 _tickTraveledScale = Vector2.One * 1.2f;

	private Tween? _actAnimTween;

	private float _mapScrollAnimTimer;

	private const string _mapTickScenePath = "res://scenes/ui/map_dot.tscn";

	private readonly double _mapAnimStartDelay = ((SaveManager.Instance.PrefsSave.FastMode == FastModeType.Fast) ? 0.5 : 1.0);

	private readonly double _mapAnimDuration = ((SaveManager.Instance.PrefsSave.FastMode == FastModeType.Fast) ? 1.5 : 3.0);

	private bool _canInterruptAnim;

	private bool _isInputDisabled;

	private RunState _runState;

	private Tween? _promptTween;

	private NMapDrawingInput? _drawingInput;

	public static NMapScreen? Instance => NRun.Instance?.GlobalUi.MapScreen;

	public bool IsOpen { get; private set; }

	public bool IsTravelEnabled { get; private set; }

	public bool IsDebugTravelEnabled { get; private set; }

	private float MapLegendX => base.Size.X * 0.8f;

	public bool IsTraveling { get; set; }

	public Dictionary<Player, MapCoord?> PlayerVoteDictionary { get; } = new Dictionary<Player, MapCoord?>();

	public NMapDrawings Drawings { get; private set; }

	public static IEnumerable<string> AssetPaths => NMapDrawings.AssetPaths.Append("res://scenes/ui/map_dot.tscn");

	public Control DefaultFocusedControl
	{
		get
		{
			NMapPoint nMapPoint = _mapPointDictionary.Values.FirstOrDefault((NMapPoint n) => n.IsEnabled);
			if (nMapPoint != null)
			{
				return nMapPoint;
			}
			return this;
		}
	}

	public event Action<MapPointType>? PointTypeHighlighted;

	public override void _Ready()
	{
		GetNode<MegaLabel>("MapLegend/Header").SetTextAutoSize(new LocString("map", "LEGEND_HEADER").GetFormattedText());
		_mapContainer = GetNode<Control>("TheMap");
		_mapBgContainer = GetNode<NMapBg>("%MapBg");
		_pathsContainer = GetNode<Control>("TheMap/Paths");
		_points = GetNode<Control>("TheMap/Points");
		_marker = GetNode<NMapMarker>("TheMap/MapMarker");
		Drawings = GetNode<NMapDrawings>("TheMap/Drawings");
		_backButton = GetNode<NBackButton>("Back");
		_backButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnBackButtonPressed));
		_backButton.Disable();
		_mapLegend = GetNode<Control>("MapLegend");
		_legendItems = GetNode<Control>("MapLegend/LegendItems");
		_legendHotkeyIcon = GetNode<TextureRect>("MapLegend/LegendHotkeyIcon");
		_drawingToolsHotkeyIcon = GetNode<TextureRect>("DrawingToolsHotkey");
		_backstop = GetNode<Control>("%Backstop");
		_drawingTools = GetNode<Control>("%DrawingTools");
		_mapDrawingButton = GetNode<NMapDrawButton>("%DrawButton");
		_mapDrawingButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnMapDrawingButtonPressed));
		_mapErasingButton = GetNode<NMapEraseButton>("%EraseButton");
		_mapErasingButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnMapErasingButtonPressed));
		_mapClearButton = GetNode<NMapClearButton>("%ClearButton");
		_mapClearButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnClearMapDrawingButtonPressed));
		RunManager.Instance.MapSelectionSynchronizer.PlayerVoteChanged += OnPlayerVoteChanged;
		RunManager.Instance.MapSelectionSynchronizer.PlayerVoteCancelled += OnPlayerVoteCancelled;
		base.ProcessMode = (ProcessModeEnum)(base.Visible ? 0 : 4);
		Connect(CanvasItem.SignalName.VisibilityChanged, Callable.From(OnVisibilityChanged));
		Callable.From(() => NCapstoneContainer.Instance.Connect(NCapstoneContainer.SignalName.Changed, Callable.From(OnCapstoneChanged))).CallDeferred();
		List<NMapLegendItem> list = _legendItems.GetChildren().OfType<NMapLegendItem>().ToList();
		for (int num = 0; num < list.Count; num++)
		{
			list[num].FocusNeighborTop = ((num > 0) ? list[num - 1].GetPath() : list[num].GetPath());
			list[num].FocusNeighborBottom = ((num < list.Count - 1) ? list[num + 1].GetPath() : list[num].GetPath());
			list[num].FocusNeighborRight = list[num].GetPath();
		}
		NControllerManager.Instance.Connect(NControllerManager.SignalName.MouseDetected, Callable.From(UpdateHotkeyDisplay));
		NControllerManager.Instance.Connect(NControllerManager.SignalName.ControllerDetected, Callable.From(UpdateHotkeyDisplay));
		NInputManager.Instance.Connect(NInputManager.SignalName.InputRebound, Callable.From(UpdateHotkeyDisplay));
		UpdateHotkeyDisplay();
	}

	public override void _ExitTree()
	{
		RunManager.Instance.MapSelectionSynchronizer.PlayerVoteChanged -= OnPlayerVoteChanged;
		RunManager.Instance.MapSelectionSynchronizer.PlayerVoteCancelled -= OnPlayerVoteCancelled;
	}

	public void Initialize(RunState runState)
	{
		_runState = runState;
		Drawings.Initialize(RunManager.Instance.NetService, _runState, RunManager.Instance.InputSynchronizer);
		_marker.Initialize(LocalContext.GetMe(_runState));
		_mapBgContainer.Initialize(_runState);
	}

	public void SetMap(ActMap map, uint seed, bool clearDrawings)
	{
		_map = map;
		_mapPointDictionary.Clear();
		_paths.Clear();
		RemoveAllMapPointsAndPaths();
		_marker.ResetMapPoint();
		if (clearDrawings)
		{
			Drawings.ClearAllLines();
		}
		_hasPlayedAnimation = false;
		int rowCount = map.GetRowCount();
		int columnCount = map.GetColumnCount();
		float num = ((map.SecondBossMapPoint != null) ? 0.9f : 1f);
		_distY = 2325f / (float)(rowCount - 1) * num;
		_distX = 1050f / (float)columnCount;
		Rng rng = new Rng(seed, $"map_jitter_{_runState.CurrentActIndex}");
		Vector2 vector = new Vector2(-500f, 740f);
		Vector2 vector2 = new Vector2(_distX, 0f - _distY);
		foreach (MapPoint allMapPoint in map.GetAllMapPoints())
		{
			NNormalMapPoint nNormalMapPoint = NNormalMapPoint.Create(allMapPoint, this, _runState);
			nNormalMapPoint.Position = new Vector2(allMapPoint.coord.col, allMapPoint.coord.row) * vector2 + vector;
			float x = rng.NextFloat(-21f, 21f);
			float y = rng.NextFloat(-25f, 25f);
			nNormalMapPoint.Position += new Vector2(x, y);
			_mapPointDictionary.Add(allMapPoint.coord, nNormalMapPoint);
			_points.AddChildSafely(nNormalMapPoint);
			nNormalMapPoint.SetAngle(Rng.Chaotic.NextGaussianFloat(0f, 8f));
		}
		_bossPointNode = NBossMapPoint.Create(map.BossMapPoint, this, _runState);
		_bossPointNode.Position = new Vector2(-200f, -1980f * num);
		_points.AddChildSafely(_bossPointNode);
		_mapPointDictionary[map.BossMapPoint.coord] = _bossPointNode;
		if (map.SecondBossMapPoint != null)
		{
			_bossPointNode.Scale = new Vector2(0.75f, 0.75f);
			_secondBossPointNode = NBossMapPoint.Create(map.SecondBossMapPoint, this, _runState);
			_secondBossPointNode.Position = new Vector2(-200f, -2280f * num);
			_secondBossPointNode.Scale = new Vector2(0.75f, 0.75f);
			_points.AddChildSafely(_secondBossPointNode);
			_mapPointDictionary[map.SecondBossMapPoint.coord] = _secondBossPointNode;
		}
		if (map.StartingMapPoint.PointType == MapPointType.Ancient)
		{
			_startingPointNode = NAncientMapPoint.Create(map.StartingMapPoint, this, _runState);
			_startingPointNode.Position = new Vector2(-80f, (float)map.StartingMapPoint.coord.row * (0f - _distY) + 720f);
		}
		else
		{
			_startingPointNode = NNormalMapPoint.Create(map.StartingMapPoint, this, _runState);
			_startingPointNode.Position = new Vector2(-80f, (float)map.StartingMapPoint.coord.row * (0f - _distY) + 800f);
		}
		_points.AddChildSafely(_startingPointNode);
		_mapPointDictionary[map.StartingMapPoint.coord] = _startingPointNode;
		foreach (MapPoint allMapPoint2 in map.GetAllMapPoints())
		{
			DrawPaths(_mapPointDictionary[allMapPoint2.coord], allMapPoint2);
		}
		DrawPaths(_startingPointNode, map.StartingMapPoint);
		DrawPaths(_bossPointNode, map.BossMapPoint);
		IReadOnlyList<MapCoord> visitedMapCoords = _runState.VisitedMapCoords;
		for (int i = 0; i < visitedMapCoords.Count - 1; i++)
		{
			if (!_paths.TryGetValue((visitedMapCoords[i], visitedMapCoords[i + 1]), out IReadOnlyList<TextureRect> value))
			{
				continue;
			}
			foreach (TextureRect item in value)
			{
				item.Modulate = _runState.Act.MapTraveledColor;
				item.Scale = _tickTraveledScale;
			}
		}
		InitMapVotes();
		RefreshAllMapPointVotes();
		for (int j = 0; j < map.GetRowCount(); j++)
		{
			IEnumerable<MapPoint> pointsInRow = map.GetPointsInRow(j);
			List<NMapPoint> list = pointsInRow.Select((MapPoint p) => _mapPointDictionary[p.coord]).ToList();
			for (int num2 = 0; num2 < list.Count; num2++)
			{
				list[num2].FocusNeighborLeft = ((num2 > 0) ? list[num2 - 1].GetPath() : list[num2].GetPath());
				list[num2].FocusNeighborRight = ((num2 < list.Count - 1) ? list[num2 + 1].GetPath() : list[num2].GetPath());
				list[num2].FocusNeighborTop = list[num2].GetPath();
				list[num2].FocusNeighborBottom = list[num2].GetPath();
			}
		}
		_startingPointNode.FocusNeighborLeft = _startingPointNode.GetPath();
		_startingPointNode.FocusNeighborRight = _startingPointNode.GetPath();
		_startingPointNode.FocusNeighborTop = _startingPointNode.GetPath();
		_startingPointNode.FocusNeighborBottom = _startingPointNode.GetPath();
		_bossPointNode.FocusNeighborLeft = _bossPointNode.GetPath();
		_bossPointNode.FocusNeighborRight = _bossPointNode.GetPath();
		_bossPointNode.FocusNeighborBottom = _bossPointNode.GetPath();
		if (_secondBossPointNode != null)
		{
			_bossPointNode.FocusNeighborTop = _secondBossPointNode.GetPath();
			_secondBossPointNode.FocusNeighborBottom = _bossPointNode.GetPath();
			_secondBossPointNode.FocusNeighborLeft = _secondBossPointNode.GetPath();
			_secondBossPointNode.FocusNeighborRight = _secondBossPointNode.GetPath();
			_secondBossPointNode.FocusNeighborTop = _secondBossPointNode.GetPath();
		}
		else
		{
			_bossPointNode.FocusNeighborTop = _bossPointNode.GetPath();
		}
		if (IsVisible())
		{
			RecalculateTravelability();
			RefreshAllPointVisuals();
		}
	}

	private void DrawPaths(NMapPoint mapPointNode, MapPoint mapPoint)
	{
		foreach (MapPoint child in mapPoint.Children)
		{
			if (!_mapPointDictionary.TryGetValue(child.coord, out NMapPoint value))
			{
				throw new InvalidOperationException($"Map point child with coord {child.coord} is not in the map point dictionary!");
			}
			Vector2 lineEndpoint = GetLineEndpoint(mapPointNode);
			Vector2 lineEndpoint2 = GetLineEndpoint(value);
			IReadOnlyList<TextureRect> value2 = CreatePath(lineEndpoint, lineEndpoint2);
			_paths.Add((mapPoint.coord, child.coord), value2);
		}
	}

	private Vector2 GetLineEndpoint(NMapPoint point)
	{
		if (point is NNormalMapPoint)
		{
			return point.Position;
		}
		return point.Position + point.Size * 0.5f;
	}

	private void RecalculateTravelability()
	{
		if (_runState.VisitedMapCoords.Any())
		{
			foreach (NMapPoint value in _mapPointDictionary.Values)
			{
				value.State = MapPointState.Untravelable;
			}
			foreach (MapCoord visitedMapCoord in _runState.VisitedMapCoords)
			{
				_mapPointDictionary[visitedMapCoord].State = MapPointState.Traveled;
			}
			IReadOnlyList<MapCoord> visitedMapCoords = _runState.VisitedMapCoords;
			MapCoord mapCoord = visitedMapCoords[visitedMapCoords.Count - 1];
			if (_secondBossPointNode != null && mapCoord == _bossPointNode.Point.coord)
			{
				_secondBossPointNode.State = MapPointState.Travelable;
				return;
			}
			if (mapCoord.row != _map.GetRowCount() - 1)
			{
				IEnumerable<MapPoint> enumerable = (_runState.Modifiers.OfType<Flight>().Any() ? _map.GetPointsInRow(mapCoord.row + 1) : _mapPointDictionary[mapCoord].Point.Children);
				{
					foreach (MapPoint item in enumerable)
					{
						_mapPointDictionary[item.coord].State = MapPointState.Travelable;
					}
					return;
				}
			}
			_bossPointNode.State = MapPointState.Travelable;
		}
		else
		{
			_startingPointNode.State = MapPointState.Travelable;
		}
	}

	private void InitMapVotes()
	{
		foreach (Player player in _runState.Players)
		{
			MapCoord? mapCoord = RunManager.Instance.MapSelectionSynchronizer.GetVote(player)?.coord;
			if (mapCoord.HasValue)
			{
				OnPlayerVoteChangedInternal(player, null, mapCoord.Value);
			}
		}
	}

	public void OnMapPointSelectedLocally(NMapPoint point)
	{
		Player me = LocalContext.GetMe(_runState);
		if (!PlayerVoteDictionary.TryGetValue(me, out var value) || value != point.Point.coord)
		{
			OnPlayerVoteChangedInternal(me, RunManager.Instance.MapSelectionSynchronizer.GetVote(me)?.coord, point.Point.coord);
			RunLocation source = new RunLocation(_runState.CurrentMapCoord, _runState.CurrentActIndex);
			MapVote value2 = new MapVote
			{
				coord = point.Point.coord,
				mapGenerationCount = RunManager.Instance.MapSelectionSynchronizer.MapGenerationCount
			};
			VoteForMapCoordAction action = new VoteForMapCoordAction(LocalContext.GetMe(_runState), source, value2);
			RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(action);
		}
		else if (_runState.Players.Count > 1)
		{
			RunManager.Instance.FlavorSynchronizer.SendMapPing(point.Point.coord);
		}
	}

	private void OnPlayerVoteChanged(Player player, MapVote? oldLocation, MapVote? newLocation)
	{
		Log.Info($"Player vote changed for {player.NetId}: {oldLocation}->{newLocation}");
		if (!LocalContext.IsMe(player))
		{
			OnPlayerVoteChangedInternal(player, oldLocation?.coord, newLocation?.coord);
		}
	}

	private void OnPlayerVoteCancelled(Player player)
	{
		Log.Info($"Player vote cancelled for {player.NetId}");
		OnPlayerVoteChangedInternal(player, PlayerVoteDictionary[player], null);
	}

	private void OnPlayerVoteChangedInternal(Player player, MapCoord? oldCoord, MapCoord? newCoord)
	{
		if (_runState.Players.Count > 1)
		{
			PlayerVoteDictionary[player] = newCoord;
			if (oldCoord.HasValue)
			{
				NMapPoint nMapPoint = _mapPointDictionary[oldCoord.Value];
				nMapPoint.VoteContainer.RefreshPlayerVotes();
			}
			else if (_runState.CurrentLocation.coord.HasValue)
			{
				NMapPoint nMapPoint2 = _mapPointDictionary[_runState.CurrentLocation.coord.Value];
				nMapPoint2.VoteContainer.RefreshPlayerVotes();
			}
			if (newCoord.HasValue)
			{
				NMapPoint nMapPoint3 = _mapPointDictionary[newCoord.Value];
				nMapPoint3.VoteContainer.RefreshPlayerVotes();
			}
			else if (_runState.CurrentLocation.coord.HasValue)
			{
				NMapPoint nMapPoint4 = _mapPointDictionary[_runState.CurrentLocation.coord.Value];
				nMapPoint4.VoteContainer.RefreshPlayerVotes();
			}
		}
	}

	public void InitMarker(MapCoord coord)
	{
		NMapPoint mapPoint = _mapPointDictionary[coord];
		_marker.SetMapPoint(mapPoint);
	}

	public async Task TravelToMapCoord(MapCoord coord)
	{
		IsTraveling = true;
		RecalculateTravelability();
		if (NCapstoneContainer.Instance.CurrentCapstoneScreen is NDeckViewScreen)
		{
			NCapstoneContainer.Instance.Close();
		}
		_marker.HideMapPoint();
		IsTravelEnabled = false;
		MapSplitVoteAnimation mapSplitVoteAnimation = new MapSplitVoteAnimation(this, _runState, _mapPointDictionary);
		await mapSplitVoteAnimation.TryPlay(coord);
		NMapPoint node = _mapPointDictionary[coord];
		node.OnSelected();
		float scaleMultiplier = 1f;
		if (node is NAncientMapPoint)
		{
			scaleMultiplier = 1.5f;
		}
		else if (node is NBossMapPoint)
		{
			scaleMultiplier = 2f;
		}
		NMapNodeSelectVfx nMapNodeSelectVfx = NMapNodeSelectVfx.Create(scaleMultiplier);
		SfxCmd.Play("event:/sfx/ui/map/map_select");
		node.AddChildSafely(nMapNodeSelectVfx);
		nMapNodeSelectVfx.Position += node.PivotOffset;
		IReadOnlyList<MapCoord> visitedMapCoords = _runState.VisitedMapCoords;
		SfxCmd.Play("event:/sfx/ui/wipe_map");
		Task fadeOutTask = TaskHelper.RunSafely(NGame.Instance.Transition.RoomFadeOut());
		if (visitedMapCoords.Any())
		{
			if (_paths.TryGetValue((visitedMapCoords[visitedMapCoords.Count - 1], node.Point.coord), out IReadOnlyList<TextureRect> value))
			{
				float waitPerTick = SaveManager.Instance.PrefsSave.FastMode switch
				{
					FastModeType.Fast => 0.3f, 
					FastModeType.Normal => 0.8f, 
					_ => 0f, 
				} / (float)value.Count;
				foreach (TextureRect tick in value)
				{
					await Cmd.Wait(waitPerTick);
					tick.Modulate = StsColors.pathDotTraveled;
					Tween tween = CreateTween();
					tween.TweenProperty(tick, "scale", _tickTraveledScale, 0.4).From(Vector2.One * 1.7f).SetEase(Tween.EaseType.Out)
						.SetTrans(Tween.TransitionType.Cubic);
				}
			}
		}
		_marker.SetMapPoint(node);
		await fadeOutTask;
		await RunManager.Instance.EnterMapCoord(coord);
		RefreshAllPointVisuals();
		PlayerVoteDictionary.Clear();
		RefreshAllMapPointVotes();
	}

	public void RefreshAllMapPointVotes()
	{
		foreach (NMapPoint value in _mapPointDictionary.Values)
		{
			value.VoteContainer.RefreshPlayerVotes();
		}
	}

	private void RemoveAllMapPointsAndPaths()
	{
		_points.FreeChildren();
		_pathsContainer.FreeChildren();
		_bossPointNode?.QueueFreeSafely();
		_secondBossPointNode?.QueueFreeSafely();
		_startingPointNode?.QueueFreeSafely();
	}

	private IReadOnlyList<TextureRect> CreatePath(Vector2 start, Vector2 end)
	{
		List<TextureRect> list = new List<TextureRect>();
		Vector2 vector = (end - start).Normalized();
		float num = vector.Angle() + (float)Math.PI / 2f;
		float num2 = start.DistanceTo(end);
		int num3 = (int)(num2 / 22f) + 1;
		for (int i = 1; i < num3; i++)
		{
			float num4 = (float)i * 22f;
			TextureRect textureRect = PreloadManager.Cache.GetScene("res://scenes/ui/map_dot.tscn").Instantiate<TextureRect>(PackedScene.GenEditState.Disabled);
			textureRect.Position = start + vector * num4;
			textureRect.Position -= new Vector2(base.Size.X * 0.5f - 20f, base.Size.Y * 0.5f - 20f);
			textureRect.Position += new Vector2(Rng.Chaotic.NextFloat(-3f, 3f), Rng.Chaotic.NextFloat(-3f, 3f));
			textureRect.FlipH = Rng.Chaotic.NextBool();
			textureRect.Rotation = num + Rng.Chaotic.NextGaussianFloat(0f, 0.1f);
			textureRect.Modulate = _runState.Act.MapUntraveledColor;
			_pathsContainer.AddChildSafely(textureRect);
			list.Add(textureRect);
		}
		return list;
	}

	public override void _Process(double delta)
	{
		if (IsVisibleInTree() && (_actAnimTween == null || !_actAnimTween.IsRunning()))
		{
			UpdateScrollPosition(delta);
		}
	}

	private void UpdateScrollPosition(double delta)
	{
		if (_mapContainer.Position != _targetDragPos)
		{
			_mapContainer.Position = _mapContainer.Position.Lerp(_targetDragPos, (float)delta * 15f);
			if (_mapContainer.Position.DistanceTo(_targetDragPos) < 0.5f)
			{
				_mapContainer.Position = _targetDragPos;
			}
		}
		if (!_isDragging)
		{
			if (_targetDragPos.Y < -600f)
			{
				_targetDragPos = _targetDragPos.Lerp(new Vector2(0f, -600f), (float)delta * 12f);
			}
			else if (_targetDragPos.Y > 1800f)
			{
				_targetDragPos = _targetDragPos.Lerp(new Vector2(0f, 1800f), (float)delta * 12f);
			}
		}
		NGame.Instance.RemoteCursorContainer.ForceUpdateAllCursors();
	}

	public override void _GuiInput(InputEvent inputEvent)
	{
		if (IsVisibleInTree())
		{
			ProcessMouseEvent(inputEvent);
			ProcessScrollEvent(inputEvent);
		}
	}

	private void ProcessMouseEvent(InputEvent inputEvent)
	{
		ProcessMouseDrawingEvent(inputEvent);
		if (_drawingInput != null)
		{
			return;
		}
		if (_isDragging && inputEvent is InputEventMouseMotion inputEventMouseMotion)
		{
			_targetDragPos += new Vector2(0f, inputEventMouseMotion.Relative.Y);
		}
		else if (inputEvent is InputEventMouseButton inputEventMouseButton)
		{
			if (inputEventMouseButton.ButtonIndex == MouseButton.Left)
			{
				if (inputEventMouseButton.Pressed && CanScroll())
				{
					_isDragging = true;
					_startDragPos = _mapContainer.Position;
					_targetDragPos = _startDragPos;
					TryCancelStartOfActAnim();
				}
				else
				{
					_isDragging = false;
				}
			}
			else if (!inputEventMouseButton.Pressed)
			{
				_isDragging = false;
			}
		}
		if (inputEvent is InputEventMouseMotion inputEventMouseMotion2 && Drawings.IsLocalDrawing())
		{
			Drawings.UpdateCurrentLinePositionLocal(Drawings.GetGlobalTransform().Inverse() * inputEventMouseMotion2.GlobalPosition);
		}
	}

	private void ProcessMouseDrawingEvent(InputEvent inputEvent)
	{
		if (!_isInputDisabled && (_actAnimTween == null || !_actAnimTween.IsRunning()) && _drawingInput == null && inputEvent is InputEventMouseButton { Pressed: not false } inputEventMouseButton)
		{
			if (inputEventMouseButton.ButtonIndex == MouseButton.Right)
			{
				_drawingInput = NMapDrawingInput.Create(Drawings, DrawingMode.Drawing, stopOnMouseRelease: true);
			}
			else if (inputEventMouseButton.ButtonIndex == MouseButton.Middle)
			{
				_drawingInput = NMapDrawingInput.Create(Drawings, DrawingMode.Erasing, stopOnMouseRelease: true);
			}
			_drawingInput?.Connect(NMapDrawingInput.SignalName.Finished, Callable.From(delegate
			{
				_drawingInput = null;
				UpdateDrawingButtonStates();
			}));
			this.AddChildSafely(_drawingInput);
		}
	}

	private void ProcessScrollEvent(InputEvent inputEvent)
	{
		if (CanScroll())
		{
			_targetDragPos += new Vector2(0f, ScrollHelper.GetDragForScrollEvent(inputEvent));
			if ((inputEvent is InputEventMouseButton || inputEvent is InputEventPanGesture) ? true : false)
			{
				TryCancelStartOfActAnim();
			}
		}
	}

	private void ProcessControllerEvent(InputEvent inputEvent)
	{
		if (inputEvent.IsActionPressed(MegaInput.up) && CanScroll())
		{
			_targetDragPos += new Vector2(0f, _controllerScrollAmount);
			TryCancelStartOfActAnim();
		}
		else if (inputEvent.IsActionPressed(MegaInput.down) && CanScroll())
		{
			_targetDragPos += new Vector2(0f, 0f - _controllerScrollAmount);
			TryCancelStartOfActAnim();
		}
		else if (inputEvent.IsActionPressed(MegaInput.right) || inputEvent.IsActionPressed(MegaInput.left) || inputEvent.IsActionPressed(MegaInput.select))
		{
			if (_runState.ActFloor == 0)
			{
				_targetDragPos = new Vector2(0f, -600f);
				return;
			}
			int num = _runState.CurrentMapCoord?.row ?? 0;
			_targetDragPos = new Vector2(0f, -600f + (float)num * _distY);
		}
	}

	public void SetTravelEnabled(bool enabled)
	{
		IsTravelEnabled = enabled && Hook.ShouldProceedToNextMapPoint(_runState);
		RefreshAllPointVisuals();
	}

	public void SetDebugTravelEnabled(bool enabled)
	{
		IsDebugTravelEnabled = enabled;
		RefreshAllPointVisuals();
	}

	public void RefreshAllPointVisuals()
	{
		foreach (NMapPoint value in _mapPointDictionary.Values)
		{
			value.RefreshVisualsInstantly();
		}
		_mapPointDictionary.Values.FirstOrDefault((NMapPoint n) => n.IsEnabled)?.TryGrabFocus();
	}

	private void PlayStartOfActAnimation()
	{
		if (_hasPlayedAnimation)
		{
			Log.Warn("Tried to play start of act animation twice! Ignoring second try");
			return;
		}
		_hasPlayedAnimation = true;
		NActBanner child = NActBanner.Create(_runState.Act, _runState.CurrentActIndex);
		NRun.Instance?.GlobalUi.MapScreen.AddChildSafely(child);
		TaskHelper.RunSafely(StartOfActAnim());
	}

	private async Task StartOfActAnim()
	{
		_mapContainer.Position = new Vector2(0f, 1800f);
		_actAnimTween?.Kill();
		_actAnimTween = CreateTween().SetParallel();
		_actAnimTween.TweenInterval(_mapAnimStartDelay);
		_actAnimTween.Chain();
		Vector2 targetDragPos = new Vector2(0f, -600f);
		_actAnimTween.TweenProperty(_mapContainer, "position:y", -600f, _mapAnimDuration).SetEase(Tween.EaseType.InOut).SetTrans(Tween.TransitionType.Expo);
		_actAnimTween.TweenCallback(Callable.From(SetInterruptable)).SetDelay(_mapAnimDuration * 0.25);
		_targetDragPos = targetDragPos;
		await ToSignal(_actAnimTween, Tween.SignalName.Finished);
		_actAnimTween = null;
		InitMapPrompt();
	}

	private void InitMapPrompt()
	{
		if (!TestMode.IsOn && !SaveManager.Instance.SeenFtue("map_select_ftue"))
		{
			TaskHelper.RunSafely(MapFtueCheck());
		}
	}

	private async Task MapFtueCheck()
	{
		await Task.Delay(100);
		NMapSelectFtue nMapSelectFtue = NMapSelectFtue.Create(_startingPointNode);
		NModalContainer.Instance.Add(nMapSelectFtue);
		SaveManager.Instance.MarkFtueAsComplete("map_select_ftue");
		await nMapSelectFtue.WaitForPlayerToConfirm();
	}

	private void SetInterruptable()
	{
		_canInterruptAnim = true;
	}

	private bool CanScroll()
	{
		if (_actAnimTween == null || _canInterruptAnim)
		{
			return !_isInputDisabled;
		}
		return false;
	}

	private void TryCancelStartOfActAnim()
	{
		if (_actAnimTween != null && _canInterruptAnim)
		{
			_actAnimTween?.Kill();
			_actAnimTween = null;
			_canInterruptAnim = false;
			_isDragging = false;
			_targetDragPos = new Vector2(0f, -600f);
			TaskHelper.RunSafely(DisableInputVeryBriefly());
		}
	}

	private async Task DisableInputVeryBriefly()
	{
		_isInputDisabled = true;
		_drawingInput?.StopDrawing();
		await Task.Delay(200);
		_isInputDisabled = false;
		InitMapPrompt();
	}

	private void OnVisibilityChanged()
	{
		if (base.Visible)
		{
			RunManager.Instance.InputSynchronizer.StartOverridingCursorPositioning(this);
			return;
		}
		_isDragging = false;
		RunManager.Instance.InputSynchronizer.StopOverridingCursorPositioning();
		_backButton.Disable();
		Drawings.StopLineLocal();
		Drawings.SetDrawingModeLocal(DrawingMode.None);
		_drawingInput?.StopDrawing();
		UpdateDrawingButtonStates();
	}

	private void OnCapstoneChanged()
	{
		_backstop.Visible = !(NCapstoneContainer.Instance?.InUse ?? false);
		if (base.Visible)
		{
			if (!_backstop.Visible)
			{
				NRun.Instance.GlobalUi.TopBar.Map.StopOscillation();
			}
			else
			{
				NRun.Instance.GlobalUi.TopBar.Map.StartOscillation();
			}
		}
	}

	public void Close(bool animateOut = true)
	{
		if (IsOpen)
		{
			IsOpen = false;
			base.FocusMode = FocusModeEnum.None;
			NRun.Instance.GlobalUi.TopBar.Map.StopOscillation();
			NHotkeyManager.Instance.RemoveHotkeyPressedBinding(MegaInput.accept, OnLegendHotkeyPressed);
			NHotkeyManager.Instance.RemoveHotkeyPressedBinding(MegaInput.viewExhaustPileAndTabRight, OnDrawingToolsHotkeyPressed);
			if (RunManager.Instance.IsSinglePlayerOrFakeMultiplayer)
			{
				CombatManager.Instance.Unpause();
			}
			_backButton.Disable();
			ActiveScreenContext.Instance.Update();
			EmitSignalClosed();
			if (animateOut)
			{
				TaskHelper.RunSafely(AnimClose());
				SfxCmd.Play("event:/sfx/ui/map/map_close");
			}
			else
			{
				base.Visible = false;
				base.ProcessMode = ProcessModeEnum.Disabled;
			}
		}
	}

	private async Task AnimClose()
	{
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_backstop, "modulate:a", 0f, 0.15);
		_tween.TweenProperty(_points, "modulate:a", 0f, 0.15);
		_tween.TweenProperty(_mapContainer, "modulate", StsColors.transparentBlack, 0.25).SetDelay(0.1);
		_tween.TweenProperty(_mapContainer, "position:y", _mapContainer.Position.Y + 200f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		_tween.TweenProperty(_mapLegend, "modulate:a", 0f, 0.15);
		_tween.TweenProperty(_mapLegend, "position:x", MapLegendX + 120f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		_tween.TweenProperty(_drawingTools, "modulate:a", 0f, 0.15);
		await ToSignal(_tween, Tween.SignalName.Finished);
		base.Visible = false;
		base.ProcessMode = ProcessModeEnum.Disabled;
	}

	public NMapScreen Open(bool isOpenedFromTopBar = false)
	{
		if (IsOpen)
		{
			return this;
		}
		IsOpen = true;
		base.Visible = true;
		_backButton.MoveToHidePosition();
		NHotkeyManager.Instance.PushHotkeyPressedBinding(MegaInput.accept, OnLegendHotkeyPressed);
		NHotkeyManager.Instance.PushHotkeyPressedBinding(MegaInput.viewExhaustPileAndTabRight, OnDrawingToolsHotkeyPressed);
		if (_runState.ActFloor > 0)
		{
			_backButton.Enable();
		}
		base.ProcessMode = ProcessModeEnum.Inherit;
		NRun.Instance.GlobalUi.TopBar.Map.StartOscillation();
		if (RunManager.Instance.IsSinglePlayerOrFakeMultiplayer)
		{
			CombatManager.Instance.Pause();
		}
		Color modulate;
		if (((_runState.CurrentActIndex == 0 && _runState.ExtraFields.StartedWithNeow) ? (_runState.ActFloor == 1) : (_runState.ActFloor == 0)) && !_hasPlayedAnimation)
		{
			if (!isOpenedFromTopBar && (SaveManager.Instance.PrefsSave.FastMode < FastModeType.Fast || !SaveManager.Instance.SeenFtue("map_select_ftue")))
			{
				PlayStartOfActAnimation();
			}
			else
			{
				_hasPlayedAnimation = true;
				Control mapContainer = _mapContainer;
				Vector2 position = _mapContainer.Position;
				position.Y = -600f;
				mapContainer.Position = position;
				_targetDragPos = new Vector2(0f, -600f);
				NActBanner child = NActBanner.Create(_runState.Act, _runState.CurrentActIndex);
				NRun.Instance.GlobalUi.MapScreen.AddChildSafely(child);
			}
		}
		else
		{
			int num = _runState.CurrentMapCoord?.row ?? 0;
			_targetDragPos = new Vector2(0f, -600f + (float)num * _distY);
			_mapContainer.Position = new Vector2(0f, -600f + (float)num * _distY);
			Control points = _points;
			modulate = _points.Modulate;
			modulate.A = 0f;
			points.Modulate = modulate;
			Control backstop = _backstop;
			modulate = _backstop.Modulate;
			modulate.A = 0f;
			backstop.Modulate = modulate;
			_mapLegend.Modulate = StsColors.transparentBlack;
			_drawingTools.Modulate = StsColors.transparentBlack;
		}
		Control mapLegend = _mapLegend;
		modulate = _mapLegend.Modulate;
		modulate.A = 0f;
		mapLegend.Modulate = modulate;
		Control drawingTools = _drawingTools;
		modulate = _drawingTools.Modulate;
		modulate.A = 0f;
		drawingTools.Modulate = modulate;
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_backstop, "modulate:a", 0.85f, 0.25);
		_tween.TweenProperty(_mapContainer, "modulate", Colors.White, 0.25).From(StsColors.transparentBlack);
		_tween.TweenProperty(_mapLegend, "modulate", Colors.White, 0.25).SetDelay(0.1);
		_tween.TweenProperty(_mapLegend, "position:x", MapLegendX, 0.25).From(MapLegendX + 120f).SetEase(Tween.EaseType.Out)
			.SetTrans(Tween.TransitionType.Back)
			.SetDelay(0.1);
		_tween.TweenProperty(_drawingTools, "modulate", Colors.White, 0.25).SetDelay(0.2);
		_tween.TweenProperty(_points, "modulate:a", 1f, 0.25).SetDelay(0.1);
		RecalculateTravelability();
		if (_runState.VisitedMapCoords.Count != 0)
		{
			IReadOnlyList<MapCoord> visitedMapCoords = _runState.VisitedMapCoords;
			MapCoord key = visitedMapCoords[visitedMapCoords.Count - 1];
			if (_bossPointNode.Point.coord.row != key.row && _startingPointNode.Point.coord.row != key.row)
			{
				NMapPoint mapPoint = _mapPointDictionary[key];
				_marker.SetMapPoint(mapPoint);
			}
		}
		SfxCmd.Play("event:/sfx/ui/map/map_open");
		ActiveScreenContext.Instance.Update();
		EmitSignalOpened();
		NMapPoint nMapPoint = _mapPointDictionary.Values.FirstOrDefault((NMapPoint n) => n.IsEnabled);
		if (nMapPoint == null)
		{
			base.FocusMode = FocusModeEnum.All;
		}
		return this;
	}

	private void OnBackButtonPressed(NButton _)
	{
		Close();
	}

	public override void _Input(InputEvent inputEvent)
	{
		if ((GetViewport().GuiGetFocusOwner() is NMapPoint || HasFocus()) && ActiveScreenContext.Instance.IsCurrent(this))
		{
			if (inputEvent.IsActionReleased(DebugHotkey.unlockCharacters))
			{
				_mapLegend.Visible = !_mapLegend.Visible;
				NGame.Instance.AddChildSafely(NFullscreenTextVfx.Create(_mapLegend.Visible ? "Show Legend" : "Hide Legend"));
			}
			if (IsVisibleInTree())
			{
				ProcessControllerEvent(inputEvent);
			}
		}
	}

	private void OnMapDrawingButtonPressed(NButton _)
	{
		NMapDrawingInput drawingInput = _drawingInput;
		if (drawingInput != null && drawingInput.DrawingMode == DrawingMode.Drawing)
		{
			_drawingInput?.StopDrawing();
		}
		else
		{
			_drawingInput?.StopDrawing();
			_drawingInput = NMapDrawingInput.Create(Drawings, DrawingMode.Drawing);
			_drawingInput.Connect(NMapDrawingInput.SignalName.Finished, Callable.From(delegate
			{
				_drawingInput = null;
				UpdateDrawingButtonStates();
			}));
			this.AddChildSafely(_drawingInput);
		}
		UpdateDrawingButtonStates();
	}

	private void OnMapErasingButtonPressed(NButton _)
	{
		NMapDrawingInput drawingInput = _drawingInput;
		if (drawingInput != null && drawingInput.DrawingMode == DrawingMode.Erasing)
		{
			_drawingInput?.StopDrawing();
		}
		else
		{
			_drawingInput?.StopDrawing();
			_drawingInput = NMapDrawingInput.Create(Drawings, DrawingMode.Erasing);
			_drawingInput.Connect(NMapDrawingInput.SignalName.Finished, Callable.From(delegate
			{
				_drawingInput = null;
				UpdateDrawingButtonStates();
			}));
			this.AddChildSafely(_drawingInput);
		}
		UpdateDrawingButtonStates();
	}

	private void UpdateDrawingButtonStates()
	{
		_mapDrawingButton.SetIsDrawing(Drawings.GetLocalDrawingMode() == DrawingMode.Drawing);
		_mapErasingButton.SetIsErasing(Drawings.GetLocalDrawingMode() == DrawingMode.Erasing);
	}

	private void OnClearMapDrawingButtonPressed(NButton _)
	{
		Drawings.ClearDrawnLinesLocal();
		SfxCmd.Play("event:/sfx/ui/map/map_erase");
		UpdateDrawingButtonStates();
	}

	public void HighlightPointType(MapPointType pointType)
	{
		this.PointTypeHighlighted?.Invoke(pointType);
	}

	public void PingMapCoord(MapCoord coord, Player player)
	{
		if (!_mapPointDictionary.TryGetValue(coord, out NMapPoint value))
		{
			Log.Error($"Someone tried to ping map coord {coord} that doesn't exist!");
			return;
		}
		NMapPingVfx nMapPingVfx = NMapPingVfx.Create();
		nMapPingVfx.Modulate = player.Character.MapDrawingColor;
		value.AddChildSafely(nMapPingVfx);
		value.MoveChild(nMapPingVfx, 0);
		nMapPingVfx.Position = Vector2.Zero;
		nMapPingVfx.Size *= value.Size.X * (1f / 64f);
		nMapPingVfx.PivotOffset = nMapPingVfx.Size * 0.5f;
		NRun.Instance.GlobalUi.MultiplayerPlayerContainer.FlashPlayerReady(player);
		NDebugAudioManager.Instance.Play("map_ping.mp3", 1f, PitchVariance.Medium);
	}

	private void OnLegendHotkeyPressed()
	{
		List<NMapLegendItem> list = _legendItems.GetChildren().OfType<NMapLegendItem>().ToList();
		if (list.Any((NMapLegendItem c) => GetViewport().GuiGetFocusOwner() == c))
		{
			_mapPointDictionary.Values.FirstOrDefault((NMapPoint n) => n.IsEnabled)?.TryGrabFocus();
			return;
		}
		NMapPoint nMapPoint = _mapPointDictionary.Values.LastOrDefault((NMapPoint n) => n.IsEnabled);
		if (nMapPoint != null)
		{
			foreach (NMapLegendItem item in list)
			{
				if (nMapPoint != null)
				{
					item.FocusNeighborLeft = nMapPoint.GetPath();
				}
				else
				{
					item.FocusNeighborLeft = GetPath();
				}
			}
		}
		list[0].TryGrabFocus();
	}

	private void OnDrawingToolsHotkeyPressed()
	{
		NMapDrawingInput drawingInput = _drawingInput;
		if (drawingInput != null && drawingInput.DrawingMode == DrawingMode.Erasing)
		{
			_mapErasingButton.TryGrabFocus();
		}
		else
		{
			_mapDrawingButton.TryGrabFocus();
		}
	}

	public Vector2 GetNetPositionFromScreenPosition(Vector2 screenPosition)
	{
		Vector2 vector = _mapBgContainer.GetGlobalTransformWithCanvas().Inverse() * screenPosition;
		Vector2 vector2 = _mapBgContainer.Size * 0.5f;
		Vector2 vector3 = new Vector2(960f, vector2.Y);
		return (vector - vector2) / vector3;
	}

	private Vector2 GetMapPositionFromNetPosition(Vector2 netPosition)
	{
		Vector2 vector = _mapBgContainer.Size * 0.5f;
		Vector2 vector2 = new Vector2(960f, vector.Y);
		return netPosition * vector2 + vector;
	}

	public Vector2 GetScreenPositionFromNetPosition(Vector2 netPosition)
	{
		Vector2 mapPositionFromNetPosition = GetMapPositionFromNetPosition(netPosition);
		return _mapBgContainer.GetGlobalTransformWithCanvas() * mapPositionFromNetPosition;
	}

	public bool IsNodeOnScreen(NMapPoint mapPoint)
	{
		float y = mapPoint.GlobalPosition.Y;
		if (y > 0f)
		{
			return y < base.Size.Y;
		}
		return false;
	}

	public void CleanUp()
	{
		if (RunManager.Instance.IsSinglePlayerOrFakeMultiplayer)
		{
			CombatManager.Instance.Unpause();
		}
	}

	private void UpdateHotkeyDisplay()
	{
		_legendHotkeyIcon.Visible = NControllerManager.Instance.IsUsingController;
		_legendHotkeyIcon.Texture = NInputManager.Instance.GetHotkeyIcon(MegaInput.accept);
		_drawingToolsHotkeyIcon.Visible = NControllerManager.Instance.IsUsingController;
		_drawingToolsHotkeyIcon.Texture = NInputManager.Instance.GetHotkeyIcon(MegaInput.viewExhaustPileAndTabRight);
	}
}
