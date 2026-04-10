using System;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Multiplayer;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Vfx;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NMouseCardPlay : NCardPlay
{
	private const float _fakeLowerEnterPlayZoneDistance = 100f;

	private const float _fakeUpperEnterPlayZoneDistance = 50f;

	private const float _playZoneScreenProportion = 0.75f;

	private const float _cancelZoneScreenProportion = 0.95f;

	private float _dragStartYPosition;

	private Creature? _target;

	private bool _isLeftMouseDown;

	private CancellationTokenSource _cancellationTokenSource;

	private Callable _onCreatureHoverCallable;

	private Callable _onCreatureUnhoverCallable;

	private bool _signalsConnected;

	private StringName _cancelShortcut;

	private bool _skipStartCardDrag;

	private float PlayZoneThreshold
	{
		get
		{
			float num = _viewport.GetVisibleRect().Size.Y * 0.75f;
			if (_skipStartCardDrag)
			{
				return num + 100f;
			}
			if (_dragStartYPosition > num)
			{
				return Mathf.Max(num, _dragStartYPosition - 100f);
			}
			return Mathf.Min(num, _dragStartYPosition - 50f);
		}
	}

	private float CancelZoneThreshold => _viewport.GetVisibleRect().Size.Y * 0.95f;

	public static NMouseCardPlay Create(NHandCardHolder holder, StringName cancelShortcut, bool wasStartedWithShortcut)
	{
		NMouseCardPlay nMouseCardPlay = new NMouseCardPlay();
		nMouseCardPlay.Holder = holder;
		nMouseCardPlay.Player = holder.CardModel.Owner;
		nMouseCardPlay._cancelShortcut = cancelShortcut;
		nMouseCardPlay._skipStartCardDrag = wasStartedWithShortcut;
		return nMouseCardPlay;
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (inputEvent is InputEventMouseButton { ButtonIndex: var buttonIndex } inputEventMouseButton)
		{
			switch (buttonIndex)
			{
			case MouseButton.Left:
				if (inputEventMouseButton.IsPressed())
				{
					_isLeftMouseDown = true;
				}
				else if (inputEventMouseButton.IsReleased())
				{
					_isLeftMouseDown = false;
				}
				break;
			case MouseButton.Right:
				if (inputEventMouseButton.IsPressed())
				{
					CancelPlayCard();
				}
				break;
			}
		}
		if (inputEvent.IsActionPressed(_cancelShortcut) || inputEvent.IsActionPressed(MegaInput.releaseCard))
		{
			CancelPlayCard();
			GetViewport()?.SetInputAsHandled();
		}
	}

	public override void Start()
	{
		_isLeftMouseDown = !_skipStartCardDrag;
		base.Holder.Hitbox.MouseFilter = Control.MouseFilterEnum.Ignore;
		_cancellationTokenSource = new CancellationTokenSource();
		_onCreatureHoverCallable = Callable.From<NCreature>(base.OnCreatureHover);
		_onCreatureUnhoverCallable = Callable.From<NCreature>(base.OnCreatureUnhover);
		TaskHelper.RunSafely(StartAsync());
	}

	public override void _EnterTree()
	{
		if (NControllerManager.Instance != null)
		{
			NControllerManager.Instance.Connect(NControllerManager.SignalName.MouseDetected, Callable.From(base.CancelPlayCard));
			NControllerManager.Instance.Connect(NControllerManager.SignalName.ControllerDetected, Callable.From(base.CancelPlayCard));
		}
	}

	public override void _ExitTree()
	{
		if (NControllerManager.Instance != null)
		{
			NControllerManager.Instance.Disconnect(NControllerManager.SignalName.MouseDetected, Callable.From(base.CancelPlayCard));
			NControllerManager.Instance.Disconnect(NControllerManager.SignalName.ControllerDetected, Callable.From(base.CancelPlayCard));
		}
		_cancellationTokenSource.Cancel();
		_cancellationTokenSource.Dispose();
		DisconnectTargetingSignals();
	}

	private async Task StartAsync()
	{
		if (base.Card == null || base.CardNode == null)
		{
			return;
		}
		await StartCardDrag();
		if (_cancellationTokenSource.IsCancellationRequested)
		{
			return;
		}
		if (!base.Card.CanPlay(out UnplayableReason reason, out AbstractModel preventer))
		{
			CannotPlayThisCardFtueCheck(base.Card);
			CancelPlayCard();
			LocString playerDialogueLine = reason.GetPlayerDialogueLine(preventer);
			if (playerDialogueLine != null)
			{
				NCombatRoom.Instance.CombatVfxContainer.AddChildSafely(NThoughtBubbleVfx.Create(playerDialogueLine.GetFormattedText(), base.Card.Owner.Creature, 1.0));
			}
			return;
		}
		base.CardNode.CardHighlight.AnimFlash();
		TargetMode targetMode = ((!_skipStartCardDrag) ? (_isLeftMouseDown ? TargetMode.ReleaseMouseToTarget : TargetMode.ClickMouseToTarget) : TargetMode.ClickMouseToTarget);
		await TargetSelection(targetMode);
		if (!_cancellationTokenSource.IsCancellationRequested)
		{
			if (!IsCardInPlayZone())
			{
				CancelPlayCard();
			}
			if (!_cancellationTokenSource.IsCancellationRequested)
			{
				TryPlayCard(_target);
			}
		}
	}

	private async Task StartCardDrag()
	{
		NDebugAudioManager.Instance?.Play("card_select.mp3", 0.5f);
		NHoverTipSet.Remove(base.Holder);
		_dragStartYPosition = _viewport.GetMousePosition().Y;
		if (!_skipStartCardDrag)
		{
			do
			{
				await LerpToMouse(base.Holder);
			}
			while (!IsCardInPlayZone() && !_cancellationTokenSource.IsCancellationRequested);
		}
	}

	private async Task TargetSelection(TargetMode targetMode)
	{
		if (base.Card != null)
		{
			TryShowEvokingOrbs();
			base.CardNode?.CardHighlight.AnimFlash();
			TargetType targetType = base.Card.TargetType;
			if ((targetType == TargetType.AnyEnemy || targetType == TargetType.AnyAlly) ? true : false)
			{
				await SingleCreatureTargeting(targetMode, base.Card.TargetType);
			}
			else
			{
				await MultiCreatureTargeting(targetMode);
			}
		}
	}

	private async Task SingleCreatureTargeting(TargetMode targetMode, TargetType targetType)
	{
		if (_cancellationTokenSource.IsCancellationRequested)
		{
			return;
		}
		CenterCard();
		NTargetManager instance = NTargetManager.Instance;
		instance.Connect(NTargetManager.SignalName.CreatureHovered, _onCreatureHoverCallable);
		instance.Connect(NTargetManager.SignalName.CreatureUnhovered, _onCreatureUnhoverCallable);
		_signalsConnected = true;
		instance.StartTargeting(targetType, base.CardNode, targetMode, () => IsCardInCancelZone() || _cancellationTokenSource.IsCancellationRequested, null);
		Node node = await instance.SelectionFinished();
		if (node != null)
		{
			Creature target;
			if (!(node is NCreature nCreature))
			{
				if (!(node is NMultiplayerPlayerState nMultiplayerPlayerState))
				{
					throw new ArgumentOutOfRangeException("target", node, null);
				}
				target = nMultiplayerPlayerState.Player.Creature;
			}
			else
			{
				target = nCreature.Entity;
			}
			_target = target;
		}
		DisconnectTargetingSignals();
	}

	private void DisconnectTargetingSignals()
	{
		if (_signalsConnected)
		{
			_signalsConnected = false;
			NTargetManager instance = NTargetManager.Instance;
			instance.Disconnect(NTargetManager.SignalName.CreatureHovered, _onCreatureHoverCallable);
			instance.Disconnect(NTargetManager.SignalName.CreatureUnhovered, _onCreatureUnhoverCallable);
		}
	}

	private async Task MultiCreatureTargeting(TargetMode targetMode)
	{
		bool isShowingTargetingVisuals = false;
		Func<bool> shouldFinishTargeting = ((targetMode == TargetMode.ReleaseMouseToTarget) ? ((Func<bool>)(() => !_isLeftMouseDown)) : ((Func<bool>)(() => _isLeftMouseDown)));
		do
		{
			if (isShowingTargetingVisuals)
			{
				if (!IsCardInPlayZone())
				{
					HideTargetingVisuals();
					isShowingTargetingVisuals = false;
				}
			}
			else if (IsCardInPlayZone())
			{
				ShowMultiCreatureTargetingVisuals();
				isShowingTargetingVisuals = true;
			}
			await LerpToMouse(base.Holder);
		}
		while (!shouldFinishTargeting() && !_cancellationTokenSource.IsCancellationRequested && !IsCardInCancelZone());
		if (!_cancellationTokenSource.IsCancellationRequested && IsCardInCancelZone())
		{
			CancelPlayCard();
		}
	}

	protected override void OnCancelPlayCard()
	{
		if (GodotObject.IsInstanceValid(this) && IsInsideTree())
		{
			base.Holder.Hitbox.MouseFilter = Control.MouseFilterEnum.Stop;
			_cancellationTokenSource.Cancel();
		}
	}

	private async Task LerpToMouse(NHandCardHolder cardHolder)
	{
		cardHolder.SetTargetPosition(_viewport.GetMousePosition());
		await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
	}

	private bool IsCardInPlayZone()
	{
		return _viewport.GetMousePosition().Y < PlayZoneThreshold;
	}

	private bool IsCardInCancelZone()
	{
		return _viewport.GetMousePosition().Y > CancelZoneThreshold;
	}
}
