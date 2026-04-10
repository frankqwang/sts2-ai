using System;
using Godot;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Runs.History;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Map;

public abstract partial class NMapPoint : NButton
{
	private MapPointState _state = MapPointState.Untravelable;

	protected IRunState _runState;

	protected Color _outlineColor = new Color(1f, 1f, 1f, 0.75f);

	protected const double _pressDownDur = 0.3;

	protected const double _unhoverAnimDur = 0.5;

	protected NSelectionReticle _controllerSelectionReticle;

	protected NMapScreen _screen;

	protected abstract Color TraveledColor { get; }

	protected abstract Color UntravelableColor { get; }

	protected abstract Color HoveredColor { get; }

	protected abstract Vector2 HoverScale { get; }

	protected abstract Vector2 DownScale { get; }

	protected override bool AllowFocusWhileDisabled => true;

	public NMultiplayerVoteContainer VoteContainer { get; set; }

	protected bool IsTravelable
	{
		get
		{
			NMapScreen screen = _screen;
			if (screen != null && screen.IsDebugTravelEnabled && !screen.IsTraveling)
			{
				return true;
			}
			if (_screen.IsTravelEnabled)
			{
				return State == MapPointState.Travelable;
			}
			return false;
		}
	}

	public MapPoint Point { get; protected set; }

	public MapPointState State
	{
		get
		{
			return _state;
		}
		set
		{
			if (value != _state)
			{
				_state = value;
				RefreshVisualsInstantly();
			}
		}
	}

	protected Color TargetColor
	{
		get
		{
			MapPointState state = State;
			if ((uint)(state - 1) <= 1u)
			{
				return TraveledColor;
			}
			return UntravelableColor;
		}
	}

	public override void _Ready()
	{
		if (GetType() != typeof(NMapPoint))
		{
			Log.Error($"{GetType()}");
			throw new InvalidOperationException("Don't call base._Ready()! Call ConnectSignals() instead.");
		}
		ConnectSignals();
	}

	protected override void ConnectSignals()
	{
		base.ConnectSignals();
		VoteContainer = GetNode<NMultiplayerVoteContainer>("%MapPointVoteContainer");
		_controllerSelectionReticle = GetNode<NSelectionReticle>("%SelectionReticle");
		VoteContainer.Initialize(ShouldDisplayPlayerVote, _runState.Players);
	}

	protected bool IsInputAllowed()
	{
		if (!_screen.IsTraveling)
		{
			return _screen.Drawings.GetLocalDrawingMode() == DrawingMode.None;
		}
		return false;
	}

	private bool ShouldDisplayPlayerVote(Player player)
	{
		if (_screen.PlayerVoteDictionary.TryGetValue(player, out var value) && value.HasValue)
		{
			return value == Point.coord;
		}
		return _runState.CurrentLocation.coord == Point.coord;
	}

	public void RefreshVisualsInstantly()
	{
		_controllerSelectionReticle.OnDeselect();
		RefreshColorInstantly();
		RefreshState();
	}

	public virtual void OnSelected()
	{
	}

	protected sealed override void OnRelease()
	{
		if (IsTravelable && (Point.coord.row != 0 || !TestMode.IsOff || SaveManager.Instance.SeenFtue("map_select_ftue")) && _screen.Drawings.GetLocalDrawingMode() == DrawingMode.None && (_screen.IsNodeOnScreen(this) || !NControllerManager.Instance.IsUsingController))
		{
			_screen.OnMapPointSelectedLocally(this);
		}
	}

	protected virtual void RefreshColorInstantly()
	{
	}

	protected virtual void RefreshState()
	{
		if (IsTravelable)
		{
			Enable();
		}
		else
		{
			Disable();
		}
	}

	protected override void OnFocus()
	{
		if (!IsInputAllowed())
		{
			return;
		}
		if (_isEnabled && NControllerManager.Instance.IsUsingController)
		{
			_controllerSelectionReticle.OnSelect();
		}
		if (_state != MapPointState.Traveled || !(_runState.CurrentLocation.coord != Point.coord) || NControllerManager.Instance.IsUsingController)
		{
			return;
		}
		MapPointHistoryEntry historyEntryFor = _runState.GetHistoryEntryFor(new RunLocation(Point.coord, _runState.CurrentActIndex));
		if (historyEntryFor != null)
		{
			int num = Point.coord.row + 1;
			for (int i = 0; i < _runState.MapPointHistory.Count - 1; i++)
			{
				num += _runState.MapPointHistory[i].Count;
			}
			NHoverTipSet tip = NHoverTipSet.CreateAndShowMapPointHistory(this, NMapPointHistoryHoverTip.Create(num, LocalContext.NetId.Value, historyEntryFor));
			Callable.From(delegate
			{
				tip.SetAlignment(this, HoverTip.GetHoverTipAlignment(this));
			}).CallDeferred();
		}
	}

	protected override void OnUnfocus()
	{
		_controllerSelectionReticle.OnDeselect();
		NHoverTipSet.Remove(this);
	}
}
