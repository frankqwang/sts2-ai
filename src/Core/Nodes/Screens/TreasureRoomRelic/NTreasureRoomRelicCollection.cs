using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.TreasureRelicPicking;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Relics;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;

public partial class NTreasureRoomRelicCollection : Control, IScreenContext
{
	private const ulong _noSelectionTimeMsec = 200uL;

	private Control _fightBackstop;

	private MegaLabel _fightLabel;

	private Tween? _emptyVfxTween;

	private NHandImageCollection _hands;

	private readonly List<NTreasureRoomRelicHolder> _multiplayerHolders = new List<NTreasureRoomRelicHolder>();

	private List<NTreasureRoomRelicHolder> _holdersInUse = new List<NTreasureRoomRelicHolder>();

	private readonly TaskCompletionSource _relicPickingBeganTaskCompletionSource = new TaskCompletionSource();

	private readonly TaskCompletionSource _relicPickingCompleteTaskCompletionSource = new TaskCompletionSource();

	private ulong _openedTicks;

	private IRunState _runState;

	private bool _isEmptyChest;

	public NTreasureRoomRelicHolder SingleplayerRelicHolder { get; private set; }

	public Control? DefaultFocusedControl
	{
		get
		{
			if (_holdersInUse.Count <= 0)
			{
				return null;
			}
			return _holdersInUse[_runState.GetPlayerSlotIndex(LocalContext.GetMe(_runState.Players))];
		}
	}

	public override void _Ready()
	{
		_fightBackstop = GetNode<Control>("%FightBackstop");
		_hands = GetNode<NHandImageCollection>("%HandsContainer");
		Control node = GetNode<Control>("Container");
		SingleplayerRelicHolder = node.GetNode<NTreasureRoomRelicHolder>("%SingleplayerRelicHolder");
		foreach (NTreasureRoomRelicHolder item in node.GetChildren().OfType<NTreasureRoomRelicHolder>())
		{
			if (item != SingleplayerRelicHolder)
			{
				_multiplayerHolders.Add(item);
			}
		}
		Control fightBackstop = _fightBackstop;
		Color modulate = _fightBackstop.Modulate;
		modulate.A = 0f;
		fightBackstop.Modulate = modulate;
		_fightBackstop.Visible = false;
		_fightLabel = GetNode<MegaLabel>("%FightLabel");
		_fightLabel.Text = new LocString("gameplay_ui", "TREASURE_FIGHT_TEXT").GetFormattedText();
		RunManager.Instance.TreasureRoomRelicSynchronizer.VotesChanged += RefreshVotes;
		RunManager.Instance.TreasureRoomRelicSynchronizer.RelicsAwarded += OnRelicsAwarded;
	}

	public override void _ExitTree()
	{
		RunManager.Instance.TreasureRoomRelicSynchronizer.VotesChanged -= RefreshVotes;
		RunManager.Instance.TreasureRoomRelicSynchronizer.RelicsAwarded -= OnRelicsAwarded;
	}

	public void Initialize(IRunState runState)
	{
		_runState = runState;
		_hands.Initialize(runState);
	}

	public void InitializeRelics()
	{
		IReadOnlyList<RelicModel> currentRelics = RunManager.Instance.TreasureRoomRelicSynchronizer.CurrentRelics;
		if (currentRelics == null || currentRelics.Count == 0)
		{
			_isEmptyChest = true;
			SingleplayerRelicHolder.Visible = false;
			foreach (NTreasureRoomRelicHolder multiplayerHolder in _multiplayerHolders)
			{
				multiplayerHolder.Visible = false;
			}
			SpawnEmptyChestVfx();
			return;
		}
		if (currentRelics.Count == 1)
		{
			SingleplayerRelicHolder.Initialize(currentRelics[0], _runState);
			SingleplayerRelicHolder.Visible = true;
			SingleplayerRelicHolder.Index = 0;
			SingleplayerRelicHolder.Connect(NClickableControl.SignalName.Released, Callable.From<NTreasureRoomRelicHolder>(delegate
			{
				PickRelic(SingleplayerRelicHolder);
			}));
			int num = 1;
			List<NTreasureRoomRelicHolder> list = new List<NTreasureRoomRelicHolder>(num);
			CollectionsMarshal.SetCount(list, num);
			Span<NTreasureRoomRelicHolder> span = CollectionsMarshal.AsSpan(list);
			int index = 0;
			span[index] = SingleplayerRelicHolder;
			_holdersInUse = list;
			{
				foreach (NTreasureRoomRelicHolder multiplayerHolder2 in _multiplayerHolders)
				{
					multiplayerHolder2.Visible = false;
				}
				return;
			}
		}
		SingleplayerRelicHolder.Visible = false;
		for (int num2 = 0; num2 < _multiplayerHolders.Count; num2++)
		{
			NTreasureRoomRelicHolder holder = _multiplayerHolders[num2];
			if (num2 < currentRelics.Count)
			{
				holder.Visible = true;
				holder.Relic.Model = currentRelics[num2];
				holder.Initialize(currentRelics[num2], _runState);
			}
			else
			{
				holder.Visible = false;
			}
			holder.Index = num2;
			holder.Connect(NClickableControl.SignalName.Released, Callable.From<NTreasureRoomRelicHolder>(delegate
			{
				PickRelic(holder);
			}));
			_holdersInUse.Add(holder);
			holder.VoteContainer.RefreshPlayerVotes();
		}
		for (int num3 = 0; num3 < _holdersInUse.Count; num3++)
		{
			_holdersInUse[num3].SetFocusMode(FocusModeEnum.All);
			_holdersInUse[num3].FocusNeighborTop = _holdersInUse[num3].GetPath();
			_holdersInUse[num3].FocusNeighborBottom = _holdersInUse[num3].GetPath();
			NTreasureRoomRelicHolder nTreasureRoomRelicHolder = _holdersInUse[num3];
			NodePath path;
			if (num3 <= 0)
			{
				List<NTreasureRoomRelicHolder> holdersInUse = _holdersInUse;
				path = holdersInUse[holdersInUse.Count - 1].GetPath();
			}
			else
			{
				path = _holdersInUse[num3 - 1].GetPath();
			}
			nTreasureRoomRelicHolder.FocusNeighborLeft = path;
			_holdersInUse[num3].FocusNeighborRight = ((num3 < _holdersInUse.Count - 1) ? _holdersInUse[num3 + 1].GetPath() : _holdersInUse[0].GetPath());
		}
		if (currentRelics.Count == 2)
		{
			_multiplayerHolders[1].Position = _multiplayerHolders[3].Position;
		}
	}

	private void SpawnEmptyChestVfx()
	{
		MegaLabel node = GetNode<MegaLabel>("%EmptyLabel");
		node.Text = new LocString("gameplay_ui", "TREASURE_EMPTY").GetFormattedText();
		node.Visible = true;
		_emptyVfxTween = CreateTween().SetParallel();
		_emptyVfxTween.TweenProperty(node, "self_modulate:a", 1f, 0.25);
		_emptyVfxTween.TweenProperty(node, "position:y", node.Position.Y, 1.0).From(node.Position.Y + 64f).SetEase(Tween.EaseType.Out)
			.SetTrans(Tween.TransitionType.Spring);
		_emptyVfxTween.Chain().TweenInterval(1.0);
		_emptyVfxTween.Chain();
		_emptyVfxTween.TweenProperty(node, "self_modulate:a", 0f, 1.0);
		GpuParticles2D node2 = GetNode<GpuParticles2D>("%SmokePuffVfx");
		node2.Emitting = true;
	}

	public void SetSelectionEnabled(bool isEnabled)
	{
		if (isEnabled)
		{
			SingleplayerRelicHolder.Enable();
			{
				foreach (NTreasureRoomRelicHolder multiplayerHolder in _multiplayerHolders)
				{
					multiplayerHolder.Enable();
				}
				return;
			}
		}
		SingleplayerRelicHolder.Disable();
		foreach (NTreasureRoomRelicHolder multiplayerHolder2 in _multiplayerHolders)
		{
			multiplayerHolder2.Disable();
		}
	}

	public Task RelicPickingBegan()
	{
		return _relicPickingBeganTaskCompletionSource.Task;
	}

	public Task RelicPickingFinished()
	{
		return _relicPickingCompleteTaskCompletionSource.Task;
	}

	public void AnimIn(Node chestVisual)
	{
		base.Visible = true;
		base.Modulate = Colors.Transparent;
		Tween tween = CreateTween().SetParallel();
		tween.TweenProperty(this, "modulate", Colors.White, 0.4);
		tween.TweenProperty(chestVisual, "modulate", StsColors.halfTransparentWhite, 0.4);
		if (_isEmptyChest)
		{
			LocalContext.GetMe(_runState)?.Relics.OfType<SilverCrucible>().FirstOrDefault()?.Flash();
			tween.TweenCallback(Callable.From(delegate
			{
				RunManager.Instance.TreasureRoomRelicSynchronizer.CompleteWithNoRelics();
			})).SetDelay(1.0);
			return;
		}
		foreach (NTreasureRoomRelicHolder holder in _holdersInUse)
		{
			holder.MouseFilter = MouseFilterEnum.Ignore;
			float num = ((_holdersInUse.Count == 1) ? 150f : 50f);
			float num2 = 0.2f + 0.2f * Rng.Chaotic.NextFloat();
			holder.Modulate = Colors.Black;
			NTreasureRoomRelicHolder nTreasureRoomRelicHolder = holder;
			Vector2 position = holder.Position;
			position.Y = holder.Position.Y + num;
			nTreasureRoomRelicHolder.Position = position;
			Tween tween2 = CreateTween().SetParallel();
			tween2.TweenProperty(holder, "modulate", Colors.White, 0.2).SetDelay(num2);
			tween2.TweenProperty(holder, "position:y", holder.Position.Y - num, 0.6).SetDelay(num2).SetEase(Tween.EaseType.Out)
				.SetTrans(Tween.TransitionType.Back);
			tween2.TweenCallback(Callable.From(() => holder.MouseFilter = MouseFilterEnum.Stop)).SetDelay((double)num2 + 0.6);
		}
		NRun.Instance.ScreenStateTracker.SetIsInSharedRelicPickingScreen(isInSharedRelicPicking: true);
		_hands.AnimateHandsIn();
	}

	public void AnimOut(Node chestVisual)
	{
		base.Modulate = Colors.White;
		Tween tween = CreateTween().Parallel();
		tween.TweenProperty(this, "modulate", StsColors.transparentWhite, 0.3);
		tween.TweenProperty(chestVisual, "modulate", Colors.White, 0.3);
		tween.TweenCallback(Callable.From(() => base.Visible = false));
		NRun.Instance.ScreenStateTracker.SetIsInSharedRelicPickingScreen(isInSharedRelicPicking: false);
	}

	private void PickRelic(NTreasureRoomRelicHolder holder)
	{
		if (Time.GetTicksMsec() - _openedTicks > 200)
		{
			RunManager.Instance.TreasureRoomRelicSynchronizer.PickRelicLocally(holder.Index);
		}
	}

	private void OnRelicsAwarded(List<RelicPickingResult> results)
	{
		TaskHelper.RunSafely(AnimateRelicAwards(results));
	}

	private async Task AnimateRelicAwards(List<RelicPickingResult> results)
	{
		foreach (NTreasureRoomRelicHolder item in _holdersInUse)
		{
			item.SetFocusMode(FocusModeEnum.None);
		}
		_relicPickingBeganTaskCompletionSource.SetResult();
		foreach (Player player in _runState.Players)
		{
			TreasureRoomRelicSynchronizer.PlayerVote playerVote = RunManager.Instance.TreasureRoomRelicSynchronizer.GetPlayerVote(player);
			if (playerVote.voteReceived && !playerVote.index.HasValue)
			{
				_hands.GetHand(player.NetId)?.SetSkipped();
			}
		}
		_hands.BeforeRelicsAwarded();
		List<Task> tasksToWait = new List<Task>();
		RelicPickingResultType? relicPickingResultType = null;
		results.Sort((RelicPickingResult r1, RelicPickingResult r2) => r1.type.CompareTo(r2.type));
		foreach (RelicPickingResult result in results)
		{
			NTreasureRoomRelicHolder holder = _holdersInUse.First((NTreasureRoomRelicHolder h) => h.Relic.Model == result.relic);
			holder.AnimateAwayVotes();
			if (relicPickingResultType.HasValue && result.type != relicPickingResultType)
			{
				await Cmd.Wait(0.5f);
			}
			if (result.type == RelicPickingResultType.FoughtOver)
			{
				holder.ZIndex = 1;
				_fightBackstop.Visible = true;
				Tween tween = CreateTween();
				tween.TweenProperty(holder, "global_position", (_fightBackstop.Size - holder.Size) * 0.5f, 0.25).SetTrans(Tween.TransitionType.Back).SetEase(Tween.EaseType.In);
				tween.TweenProperty(_fightBackstop, "modulate:a", 1f, 0.25);
				_hands.BeforeFightStarted(result.fight.playersInvolved);
				await ToSignal(tween, Tween.SignalName.Finished);
				await Cmd.Wait(1f);
				await _hands.DoFight(result, holder);
				tween = CreateTween();
				tween.TweenProperty(_fightBackstop, "modulate:a", 0f, 0.25);
				await ToSignal(tween, Tween.SignalName.Finished);
				_fightBackstop.Visible = false;
				holder.ZIndex = 0;
			}
			else if (result.type != RelicPickingResultType.Skipped)
			{
				NHandImage hand = _hands.GetHand(result.player.NetId);
				if (hand != null)
				{
					tasksToWait.Add(TaskHelper.RunSafely(hand.GrabRelic(holder)));
					await Cmd.Wait(0.25f);
				}
			}
			relicPickingResultType = result.type;
		}
		await Task.WhenAll(tasksToWait);
		if (tasksToWait.Count == 0 && _runState.Players.Count > 1)
		{
			await Cmd.Wait(0.7f);
		}
		foreach (RelicPickingResult result2 in results)
		{
			NTreasureRoomRelicHolder nTreasureRoomRelicHolder = _holdersInUse.First((NTreasureRoomRelicHolder h) => h.Relic.Model == result2.relic);
			RelicModel relic = result2.relic.ToMutable();
			nTreasureRoomRelicHolder.Disable();
			if (result2.type != RelicPickingResultType.Skipped)
			{
				TaskHelper.RunSafely(RelicCmd.Obtain(relic, result2.player));
				if (LocalContext.IsMe(result2.player))
				{
					NRun.Instance.GlobalUi.RelicInventory.AnimateRelic(relic, nTreasureRoomRelicHolder.GlobalPosition, nTreasureRoomRelicHolder.Scale);
				}
				if (_runState.Players.Count == 1)
				{
					nTreasureRoomRelicHolder.Visible = false;
				}
			}
			foreach (Player player2 in _runState.Players)
			{
				if (player2 != result2.player)
				{
					player2.RelicGrabBag.MoveToFallback(result2.relic);
				}
			}
		}
		_relicPickingCompleteTaskCompletionSource.SetResult();
	}

	private void RefreshVotes()
	{
		foreach (NTreasureRoomRelicHolder item in _holdersInUse)
		{
			item.VoteContainer.RefreshPlayerVotes();
		}
	}
}
