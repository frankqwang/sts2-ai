using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Relics;
using MegaCrit.Sts2.Core.Entities.TreasureRelicPicking;
using MegaCrit.Sts2.Core.Extensions;
using MegaCrit.Sts2.Core.Factories;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Relics;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Runs.History;
using MegaCrit.Sts2.Core.Unlocks;

namespace MegaCrit.Sts2.Core.Multiplayer.Game;

public class TreasureRoomRelicSynchronizer
{
	public class PlayerVote
	{
		public int? index;

		public bool voteReceived;
	}

	private readonly IPlayerCollection _playerCollection;

	private readonly ulong _localPlayerId;

	private readonly RelicGrabBag _sharedGrabBag;

	private readonly ActionQueueSynchronizer _actionQueueSynchronizer;

	private readonly Rng _rng;

	private readonly Logger _logger = new Logger("TreasureRoomRelicSynchronizer", LogType.GameSync);

	private List<RelicModel>? _currentRelics;

	private readonly List<PlayerVote> _votes = new List<PlayerVote>();

	private PlayerVote? _predictedVote;

	private bool _singlePlayerSkipped;

	public IReadOnlyList<RelicModel>? CurrentRelics => _currentRelics;

	private Player LocalPlayer => _playerCollection.GetPlayer(_localPlayerId);

	public event Action? VotesChanged;

	public event Action<List<RelicPickingResult>>? RelicsAwarded;

	public TreasureRoomRelicSynchronizer(IPlayerCollection playerCollection, ulong localPlayerId, ActionQueueSynchronizer actionQueueSynchronizer, RelicGrabBag sharedGrabBag, Rng rng)
	{
		_playerCollection = playerCollection;
		_localPlayerId = localPlayerId;
		_actionQueueSynchronizer = actionQueueSynchronizer;
		_sharedGrabBag = sharedGrabBag;
		_rng = rng;
	}

	public void BeginRelicPicking()
	{
		if (CurrentRelics != null)
		{
			throw new InvalidOperationException("Attempted to start new relic picking session while one was already occurring!");
		}
		_currentRelics = new List<RelicModel>();
		_votes.Clear();
		_predictedVote = null;
		foreach (Player player in _playerCollection.Players)
		{
			_votes.Add(new PlayerVote
			{
				voteReceived = false
			});
			IRunState runState = player.RunState;
			if (Hook.ShouldGenerateTreasure(runState, player))
			{
				RelicRarity rarity = RelicFactory.RollRarity(_rng);
				RelicModel item = TryGetRelicForTutorial(runState.UnlockState) ?? _sharedGrabBag.PullFromFront(rarity, runState) ?? RelicFactory.FallbackRelic;
				_currentRelics.Add(item);
			}
		}
		if (_currentRelics.Count > 0)
		{
			if (RunManager.Instance.IsSinglePlayerOrFakeMultiplayer && _playerCollection.Players.Count > 1)
			{
				foreach (Player player2 in _playerCollection.Players)
				{
					if (player2 != LocalPlayer)
					{
						PlayerVote playerVote = _votes[_playerCollection.GetPlayerSlotIndex(player2)];
						playerVote.index = _rng.NextInt(_currentRelics.Count);
						playerVote.voteReceived = true;
					}
				}
			}
			this.VotesChanged?.Invoke();
		}
		else
		{
			EndRelicVoting();
		}
	}

	public void SkipRelicLocally()
	{
		PickRelicLocally(null);
	}

	public void PickRelicLocally(int? index)
	{
		if (index.HasValue)
		{
			_logger.Debug($"Relic index {index} ({_currentRelics?[index.Value]}) is being picked by local player {LocalPlayer.NetId}");
		}
		else
		{
			_logger.Debug($"Relic has been skipped by local player {LocalPlayer.NetId}");
		}
		if (_currentRelics == null)
		{
			throw new InvalidOperationException("Attempted to pick relic while relic picking is not active!");
		}
		_predictedVote = new PlayerVote
		{
			index = index,
			voteReceived = true
		};
		_actionQueueSynchronizer.RequestEnqueue(new PickRelicAction(LocalPlayer, index));
		this.VotesChanged?.Invoke();
	}

	public void OnPicked(Player player, int? index)
	{
		if (index.HasValue)
		{
			_logger.Debug($"Player {player} picked relic at index {index}: {_currentRelics?[index.Value]}");
		}
		else
		{
			_logger.Debug($"Player {player} skipped relic");
		}
		if (_currentRelics == null)
		{
			_logger.Warn("Attempted to pick relic while relic picking is not active!");
			return;
		}
		if (index >= _currentRelics.Count)
		{
			throw new IndexOutOfRangeException($"Attempted to pick relic at index {index}, but there are only {_currentRelics.Count} to choose from!");
		}
		if (!index.HasValue && _playerCollection.Players.Count == 1)
		{
			_singlePlayerSkipped = true;
			return;
		}
		PlayerVote playerVote = _votes[_playerCollection.GetPlayerSlotIndex(player)];
		playerVote.index = index;
		playerVote.voteReceived = true;
		this.VotesChanged?.Invoke();
		if (!_votes.All((PlayerVote v) => v.voteReceived))
		{
			return;
		}
		if (_predictedVote != null)
		{
			PlayerVote predictedVote = _predictedVote;
			_predictedVote = null;
			if (_votes[_playerCollection.GetPlayerSlotIndex(LocalPlayer)].index != predictedVote.index)
			{
				this.VotesChanged?.Invoke();
			}
		}
		AwardRelics();
		EndRelicVoting();
	}

	private void AwardRelics()
	{
		Dictionary<int, List<Player>> dictionary = new Dictionary<int, List<Player>>();
		for (int i = 0; i < _currentRelics.Count; i++)
		{
			dictionary[i] = new List<Player>();
		}
		for (int j = 0; j < _votes.Count; j++)
		{
			Player item = _playerCollection.Players[j];
			PlayerVote playerVote = _votes[j];
			if (playerVote.index.HasValue)
			{
				List<Player> valueOrDefault = dictionary.GetValueOrDefault(playerVote.index.Value, new List<Player>());
				valueOrDefault.Add(item);
				dictionary[playerVote.index.Value] = valueOrDefault;
			}
		}
		List<RelicPickingResult> results = new List<RelicPickingResult>();
		List<RelicModel> list = new List<RelicModel>();
		foreach (KeyValuePair<int, List<Player>> item2 in dictionary)
		{
			RelicModel relicModel = _currentRelics[item2.Key];
			if (item2.Value.Count == 0)
			{
				list.Add(relicModel);
			}
			else if (item2.Value.Count == 1)
			{
				results.Add(new RelicPickingResult
				{
					type = RelicPickingResultType.OnlyOnePlayerVoted,
					relic = relicModel,
					player = item2.Value[0]
				});
			}
			else if (item2.Value.Count > 1)
			{
				RelicPickingFightMove[] possibleMoves = Enum.GetValues<RelicPickingFightMove>();
				results.Add(RelicPickingResult.GenerateRelicFight(item2.Value, relicModel, () => _rng.NextItem(possibleMoves)));
			}
		}
		List<Player> list2 = _playerCollection.Players.Where(delegate(Player p)
		{
			bool flag = results.Find((RelicPickingResult r) => r.player == p) != null;
			PlayerVote playerVote2 = _votes[_playerCollection.GetPlayerSlotIndex(p)];
			bool flag2 = playerVote2.voteReceived && !playerVote2.index.HasValue;
			return !flag && !flag2;
		}).ToList();
		list.StableShuffle(_rng);
		for (int num = 0; num < list.Count; num++)
		{
			if (num < list2.Count)
			{
				results.Add(new RelicPickingResult
				{
					type = RelicPickingResultType.ConsolationPrize,
					player = list2[num],
					relic = list[num]
				});
			}
			else
			{
				results.Add(new RelicPickingResult
				{
					type = RelicPickingResultType.Skipped,
					player = null,
					relic = list[num]
				});
			}
		}
		this.RelicsAwarded?.Invoke(results);
	}

	public void OnRoomExited()
	{
		if (_singlePlayerSkipped)
		{
			EndRelicVoting();
		}
	}

	private void EndRelicVoting()
	{
		_currentRelics = null;
		_singlePlayerSkipped = false;
	}

	public PlayerVote GetPlayerVote(Player player)
	{
		if (player == LocalPlayer && _predictedVote != null)
		{
			return _predictedVote;
		}
		return _votes[_playerCollection.GetPlayerSlotIndex(player)];
	}

	public void CompleteWithNoRelics()
	{
		_logger.Debug("Completing relic picking with no relics (empty chest)");
		this.RelicsAwarded?.Invoke(new List<RelicPickingResult>());
		_currentRelics = null;
	}

	private RelicModel? TryGetRelicForTutorial(UnlockState unlockState)
	{
		if (unlockState.NumberOfRuns == 0 && LocalContext.GetMe(_playerCollection).RunState.MapPointHistory.SelectMany((IReadOnlyList<MapPointHistoryEntry> l) => l).Count((MapPointHistoryEntry p) => p.HasRoomOfType(RoomType.Treasure)) == 1)
		{
			Log.Info("Forcing specific relic because it's the player's first treasure chest ever");
			_sharedGrabBag.Remove<Gorget>();
			return ModelDb.Relic<Gorget>();
		}
		return null;
	}
}
