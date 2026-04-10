using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Debug;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Extensions;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Multiplayer;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Relics;

public partial class NRelicInventory : FlowContainer
{
	[Signal]
	public delegate void RelicsChangedEventHandler();

	private Player? _player;

	private readonly List<NRelicInventoryHolder> _relicNodes = new List<NRelicInventoryHolder>();

	private Vector2 _originalPos;

	private Tween? _curTween;

	private Tween? _debugHideTween;

	private bool _isDebugHidden;

	public IReadOnlyList<NRelicInventoryHolder> RelicNodes => _relicNodes;

	public override void _Ready()
	{
		_originalPos = base.Position;
		Connect(Control.SignalName.FocusEntered, Callable.From(delegate
		{
			_relicNodes[0].TryGrabFocus();
		}));
	}

	public override void _EnterTree()
	{
		base._EnterTree();
		ActiveScreenContext.Instance.Updated += UpdateNavigation;
		ConnectPlayerEvents();
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		ActiveScreenContext.Instance.Updated -= UpdateNavigation;
		DisconnectPlayerEvents();
	}

	public void Initialize(RunState runState)
	{
		DisconnectPlayerEvents();
		_player = LocalContext.GetMe(runState);
		ConnectPlayerEvents();
		foreach (RelicModel relic in _player.Relics)
		{
			Add(relic, startsShown: true);
		}
	}

	private void ConnectPlayerEvents()
	{
		if (_player != null)
		{
			_player.RelicObtained += OnRelicObtained;
			_player.RelicRemoved += OnRelicRemoved;
		}
	}

	private void DisconnectPlayerEvents()
	{
		if (_player != null)
		{
			_player.RelicObtained -= OnRelicObtained;
			_player.RelicRemoved -= OnRelicRemoved;
		}
	}

	private void Add(RelicModel relic, bool startsShown, int index = -1)
	{
		NRelicInventoryHolder nRelicInventoryHolder = NRelicInventoryHolder.Create(relic);
		nRelicInventoryHolder.Inventory = this;
		if (index < 0)
		{
			_relicNodes.Add(nRelicInventoryHolder);
		}
		else
		{
			_relicNodes.Insert(index, nRelicInventoryHolder);
		}
		this.AddChildSafely(nRelicInventoryHolder);
		MoveChild(nRelicInventoryHolder, index);
		if (!startsShown)
		{
			TextureRect icon = nRelicInventoryHolder.Relic.Icon;
			Color modulate = nRelicInventoryHolder.Relic.Icon.Modulate;
			modulate.A = 0f;
			icon.Modulate = modulate;
			UpdateNavigation();
		}
		nRelicInventoryHolder.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(delegate
		{
			OnRelicClicked(relic);
		}));
		nRelicInventoryHolder.Connect(NClickableControl.SignalName.Focused, Callable.From<NClickableControl>(delegate
		{
			OnRelicFocused(relic);
		}));
		nRelicInventoryHolder.Connect(NClickableControl.SignalName.Unfocused, Callable.From<NClickableControl>(delegate
		{
			OnRelicUnfocused();
		}));
		EmitSignal(SignalName.RelicsChanged);
	}

	private void Remove(RelicModel relic)
	{
		if (LocalContext.IsMine(relic))
		{
			NRelicInventoryHolder nRelicInventoryHolder = _relicNodes.First((NRelicInventoryHolder n) => n.Relic.Model == relic);
			_relicNodes.Remove(nRelicInventoryHolder);
			this.RemoveChildSafely(nRelicInventoryHolder);
			EmitSignalRelicsChanged();
			UpdateNavigation();
		}
	}

	private void OnRelicClicked(RelicModel model)
	{
		List<RelicModel> list = new List<RelicModel>();
		foreach (NRelicInventoryHolder relicNode in _relicNodes)
		{
			list.Add(relicNode.Relic.Model);
		}
		NGame.Instance.GetInspectRelicScreen().Open(list, model);
	}

	private void OnRelicFocused(RelicModel model)
	{
		RunManager.Instance.HoveredModelTracker.OnLocalRelicHovered(model);
	}

	private static void OnRelicUnfocused()
	{
		RunManager.Instance.HoveredModelTracker.OnLocalRelicUnhovered();
	}

	public void AnimateRelic(RelicModel relic, Vector2? startPosition = null, Vector2? startScale = null)
	{
		if (LocalContext.IsMine(relic))
		{
			NRelicInventoryHolder nRelicInventoryHolder = _relicNodes.First((NRelicInventoryHolder n) => n.Relic.Model == relic);
			TaskHelper.RunSafely(nRelicInventoryHolder.PlayNewlyAcquiredAnimation(startPosition, startScale));
		}
	}

	private void OnRelicObtained(RelicModel relic)
	{
		Add(relic, startsShown: false, _player.Relics.IndexOf(relic));
	}

	private void OnRelicRemoved(RelicModel relic)
	{
		Remove(relic);
	}

	public void AnimShow()
	{
		base.FocusBehaviorRecursive = FocusBehaviorRecursiveEnum.Enabled;
		_curTween?.Kill();
		_curTween = CreateTween();
		_curTween.TweenProperty(this, "global_position:y", _originalPos.Y, 0.25).SetTrans(Tween.TransitionType.Cubic).SetEase(Tween.EaseType.Out);
	}

	public void AnimHide()
	{
		base.FocusBehaviorRecursive = FocusBehaviorRecursiveEnum.Disabled;
		_curTween?.Kill();
		_curTween = CreateTween();
		_curTween.TweenProperty(this, "global_position:y", _originalPos.Y - 68f * (float)GetLineCount() - 90f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
	}

	public void ShowImmediately()
	{
		_curTween?.Kill();
		Vector2 position = base.Position;
		position.Y = _originalPos.Y;
		base.Position = position;
	}

	public void HideImmediately()
	{
		_curTween?.Kill();
		Vector2 position = base.Position;
		position.Y = _originalPos.Y - 68f * (float)GetLineCount() - 90f;
		base.Position = position;
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (inputEvent.IsActionReleased(DebugHotkey.hideTopBar))
		{
			DebugHideTopBar();
		}
	}

	private void DebugHideTopBar()
	{
		if (_isDebugHidden)
		{
			AnimShow();
		}
		else
		{
			AnimHide();
		}
		_isDebugHidden = !_isDebugHidden;
	}

	private void UpdateNavigation()
	{
		for (int i = 0; i < RelicNodes.Count; i++)
		{
			NRelicInventoryHolder nRelicInventoryHolder = RelicNodes[i];
			nRelicInventoryHolder.FocusNeighborLeft = ((i > 0) ? RelicNodes[i - 1].GetPath() : RelicNodes[i].GetPath());
			nRelicInventoryHolder.FocusNeighborRight = ((i < RelicNodes.Count - 1) ? RelicNodes[i + 1].GetPath() : RelicNodes[i].GetPath());
			Control firstPotionControl = NRun.Instance.GlobalUi.TopBar.PotionContainer.FirstPotionControl;
			nRelicInventoryHolder.FocusNeighborTop = ((firstPotionControl != null && GodotObject.IsInstanceValid(firstPotionControl)) ? firstPotionControl.GetPath() : null);
			NMultiplayerPlayerStateContainer multiplayerPlayerContainer = NRun.Instance.GlobalUi.MultiplayerPlayerContainer;
			if (multiplayerPlayerContainer.GetChildCount() > 0)
			{
				Control control = multiplayerPlayerContainer.FirstPlayerState?.Hitbox;
				nRelicInventoryHolder.FocusNeighborBottom = ((control != null && GodotObject.IsInstanceValid(control)) ? control.GetPath() : null);
			}
			else
			{
				Control control2 = ActiveScreenContext.Instance.GetCurrentScreen()?.FocusedControlFromTopBar;
				nRelicInventoryHolder.FocusNeighborBottom = ((control2 != null && GodotObject.IsInstanceValid(control2)) ? control2.GetPath() : null);
			}
		}
	}
}
