using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.TreasureRelicPicking;
using MegaCrit.Sts2.Core.Extensions;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Multiplayer.Game.PeerInput;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;

public partial class NHandImageCollection : Control
{
	private PeerInputSynchronizer? _synchronizer;

	private readonly List<NHandImage> _hands = new List<NHandImage>();

	private IRunState _runState = NullRunState.Instance;

	private float _handAnimateInProgress;

	public override void _EnterTree()
	{
		base._EnterTree();
		GetViewport().Connect(Viewport.SignalName.GuiFocusChanged, Callable.From<Control>(ProcessGuiFocus));
	}

	public override void _ExitTree()
	{
		if (_synchronizer != null)
		{
			_synchronizer.StateAdded -= OnInputStateAdded;
			_synchronizer.StateChanged -= OnInputStateChanged;
			_synchronizer.StateRemoved -= OnInputStateRemoved;
		}
		NGame.Instance.CursorManager.SetCursorShown(show: true);
		GetViewport().Disconnect(Viewport.SignalName.GuiFocusChanged, Callable.From<Control>(ProcessGuiFocus));
	}

	public void Initialize(IRunState runState)
	{
		_runState = runState;
		if (_runState.Players.Count <= 1)
		{
			return;
		}
		_synchronizer = RunManager.Instance.InputSynchronizer;
		_synchronizer.StateAdded -= OnInputStateAdded;
		_synchronizer.StateChanged += OnInputStateChanged;
		_synchronizer.StateRemoved += OnInputStateRemoved;
		foreach (Player player in _runState.Players)
		{
			AddHand(player.NetId);
		}
		UpdateHandVisibility();
	}

	private void OnInputStateAdded(ulong playerId)
	{
		AddHand(playerId);
	}

	private void OnInputStateRemoved(ulong playerId)
	{
		RemoveHand(playerId);
	}

	private void AddHand(ulong playerId)
	{
		if (_hands.Any((NHandImage c) => c.Player.NetId == playerId))
		{
			Log.Error($"Tried to add hand for player {playerId} twice!");
		}
		else
		{
			Player player = _runState.GetPlayer(playerId);
			NHandImage nHandImage = NHandImage.Create(player, _runState.GetPlayerSlotIndex(player));
			_hands.Add(nHandImage);
			this.AddChildSafely(nHandImage);
		}
	}

	private void OnInputStateChanged(ulong playerId)
	{
		Vector2 controlSpaceFocusPosition = _synchronizer.GetControlSpaceFocusPosition(playerId, this);
		NHandImage hand = GetHand(playerId);
		hand.SetIsDown(_synchronizer.GetMouseDown(playerId));
		hand.SetPointingPosition(controlSpaceFocusPosition);
		UpdateHandVisibility();
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (_runState.Players.Count == 1)
		{
			return;
		}
		if (inputEvent is InputEventMouseMotion)
		{
			NHandImage hand = GetHand(LocalContext.NetId.Value);
			hand.SetPointingPosition(GetGlobalMousePosition());
		}
		else if (inputEvent is InputEventMouseButton inputEventMouseButton && inputEventMouseButton.ButtonIndex == MouseButton.Left)
		{
			NHandImage hand2 = GetHand(LocalContext.NetId.Value);
			if (inputEventMouseButton.IsPressed() && !hand2.IsDown)
			{
				hand2.SetIsDown(isDown: true);
			}
			else if (inputEventMouseButton.IsReleased() && hand2.IsDown)
			{
				hand2.SetIsDown(isDown: false);
			}
		}
	}

	private void ProcessGuiFocus(Control focusedControl)
	{
		if (IsVisibleInTree() && NControllerManager.Instance.IsUsingController && _runState.Players.Count != 1)
		{
			if (focusedControl is NTreasureRoomRelicHolder)
			{
				NHandImage hand = GetHand(LocalContext.NetId.Value);
				Vector2 vector = Vector2.Down.Rotated(hand.Rotation);
				Vector2 pointingPosition = focusedControl.GlobalPosition + focusedControl.Size * 0.5f + vector * 100f;
				hand.SetPointingPosition(pointingPosition);
			}
			else
			{
				NHandImage hand2 = GetHand(LocalContext.NetId.Value);
				hand2.AnimateAway();
			}
		}
	}

	public NHandImage? GetHand(ulong playerId)
	{
		return _hands.FirstOrDefault((NHandImage c) => c.Player.NetId == playerId);
	}

	private void RemoveHand(ulong playerId)
	{
		NHandImage hand = GetHand(playerId);
		if (hand != null)
		{
			hand.QueueFreeSafely();
			_hands.Remove(hand);
		}
	}

	private void UpdateHandVisibility()
	{
		NetScreenType screenType = _synchronizer.GetScreenType(LocalContext.NetId.Value);
		foreach (NHandImage hand in _hands)
		{
			NetScreenType netScreenType = ((!RunManager.Instance.IsSinglePlayerOrFakeMultiplayer) ? _synchronizer.GetScreenType(hand.Player.NetId) : NetScreenType.SharedRelicPicking);
			bool flag = netScreenType == NetScreenType.SharedRelicPicking && screenType == NetScreenType.SharedRelicPicking;
			if (!hand.Visible && flag)
			{
				hand.AnimateIn();
			}
			hand.Visible = flag;
		}
		NGame.Instance.CursorManager.SetCursorShown(screenType != NetScreenType.SharedRelicPicking);
	}

	public void BeforeRelicsAwarded()
	{
		foreach (NHandImage hand in _hands)
		{
			hand.SetFrozenForRelicAwards(frozenForRelicAwards: true);
		}
	}

	public void BeforeFightStarted(List<Player> playersInvolved)
	{
		foreach (Player item in playersInvolved)
		{
			NHandImage hand = GetHand(item.NetId);
			hand.SetIsInFight(inFight: true);
		}
	}

	public void AnimateHandsIn()
	{
		foreach (NHandImage hand in _hands)
		{
			if (hand.Visible)
			{
				hand.AnimateIn();
			}
		}
	}

	public void AnimateHandsAway()
	{
		foreach (NHandImage hand in _hands)
		{
			hand.AnimateAway();
		}
	}

	public async Task DoFight(RelicPickingResult result, NTreasureRoomRelicHolder holder)
	{
		RelicPickingFight fight = result.fight;
		List<Tween> tweens = new List<Tween>();
		List<Task> tasks = new List<Task>();
		for (int i = 0; i < fight.rounds.Count; i++)
		{
			float durationMultiplier = 1.5f / ((float)i + 1.5f);
			RelicPickingFightRound round = fight.rounds[i];
			tweens.Clear();
			for (int j = 0; j < fight.playersInvolved.Count; j++)
			{
				RelicPickingFightMove? relicPickingFightMove = round.moves[j];
				if (relicPickingFightMove.HasValue)
				{
					Player player = fight.playersInvolved[j];
					NHandImage hand = GetHand(player.NetId);
					tweens.Add(hand.DoFightMove(relicPickingFightMove.Value, 1.5f * durationMultiplier));
				}
			}
			await Task.WhenAll(tweens.Select((Tween t) => ToSignal(t, Tween.SignalName.Finished).ToTask()));
			List<Player> list = new List<Player>();
			for (int num = 0; num < fight.playersInvolved.Count; num++)
			{
				Player player2 = fight.playersInvolved[num];
				if (i < fight.rounds.Count - 1)
				{
					if (round.moves[num].HasValue && !fight.rounds[i + 1].moves[num].HasValue)
					{
						list.Add(player2);
					}
				}
				else if (round.moves[num].HasValue && result.player != player2)
				{
					list.Add(player2);
				}
			}
			tasks.Clear();
			foreach (Player item in list)
			{
				tasks.Add(DoLoseShake(item, Mathf.Max(1f * durationMultiplier, 0.5f)));
			}
			if (i == fight.rounds.Count - 1)
			{
				await Cmd.Wait(0.5f);
				NHandImage hand2 = GetHand(result.player.NetId);
				tasks.Add(hand2.GrabRelic(holder));
			}
			if (tasks.Count == 0)
			{
				await Cmd.Wait(1f * durationMultiplier);
			}
			else
			{
				await Task.WhenAll(tasks);
			}
		}
		foreach (Player item2 in fight.playersInvolved)
		{
			NHandImage hand3 = GetHand(item2.NetId);
			hand3.SetIsInFight(inFight: false);
		}
	}

	private async Task DoLoseShake(Player player, float duration)
	{
		NHandImage hand = GetHand(player.NetId);
		await hand.DoLoseShake(duration);
		hand.SetIsInFight(inFight: false);
	}
}
