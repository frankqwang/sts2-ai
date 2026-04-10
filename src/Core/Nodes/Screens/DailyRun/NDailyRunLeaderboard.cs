using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Daily;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Leaderboard;
using MegaCrit.Sts2.Core.Localization;

namespace MegaCrit.Sts2.Core.Nodes.Screens.DailyRun;

public partial class NDailyRunLeaderboard : Control
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/daily_run/daily_run_leaderboard");

	private const int _maxEntries = 10;

	private MegaLabel _titleLabel;

	private NLeaderboardDayPaginator _paginator;

	private VBoxContainer _scoreContainer;

	private NLeaderboardPageArrow _leftArrow;

	private NLeaderboardPageArrow _rightArrow;

	private MegaRichTextLabel _loadingIndicator;

	private MegaLabel _noScoresIndicator;

	private MegaLabel _noFriendsIndicator;

	private Control? _noScoreUploadIndicator;

	private int _currentPage;

	private DateTimeOffset _todaysDailyTime;

	private DateTimeOffset _leaderboardTime;

	private readonly List<ulong> _playersInRun = new List<ulong>();

	private bool _hasNegativeScore;

	private CancellationTokenSource? _loadCts;

	private static readonly LocString _titleLoc = new LocString("main_menu_ui", "DAILY_RUN_MENU.LEADERBOARDS.title");

	private static readonly LocString _scoreLoc = new LocString("main_menu_ui", "DAILY_RUN_MENU.LEADERBOARDS.noScore");

	private static readonly LocString _fetchingScoreLoc = new LocString("main_menu_ui", "DAILY_RUN_MENU.LEADERBOARDS.fetchingScores");

	private static readonly LocString _friendsLoc = new LocString("main_menu_ui", "DAILY_RUN_MENU.LEADERBOARDS.noFriends");

	public static string[] AssetPaths => new string[1] { _scenePath };

	public override void _Ready()
	{
		_titleLabel = GetNode<MegaLabel>("%Title");
		_paginator = GetNode<NLeaderboardDayPaginator>("Paginator");
		_scoreContainer = GetNodeOrNull<VBoxContainer>("%ScoreContainer") ?? GetNodeOrNull<VBoxContainer>("%LeaderboardScoreContainer") ?? throw new InvalidOperationException("Couldn't find score container");
		_leftArrow = GetNode<NLeaderboardPageArrow>("%LeftArrow");
		_rightArrow = GetNode<NLeaderboardPageArrow>("%RightArrow");
		_loadingIndicator = GetNode<MegaRichTextLabel>("%LoadingText");
		_noScoresIndicator = GetNode<MegaLabel>("%NoScoresIndicator");
		_noFriendsIndicator = GetNode<MegaLabel>("%NoFriendsIndicator");
		_noScoreUploadIndicator = GetNodeOrNull<Control>("%ScoreWarning");
		CallDeferred(MethodName.SetLocalizedText);
		_loadingIndicator.SetTextAutoSize(_fetchingScoreLoc.GetFormattedText());
		_rightArrow.Connect(delegate
		{
			ChangePage(1);
		});
		_leftArrow.Connect(delegate
		{
			ChangePage(-1);
		});
	}

	private void SetLocalizedText()
	{
		_titleLabel.SetTextAutoSize(_titleLoc.GetFormattedText());
		_noScoresIndicator.SetTextAutoSize(_scoreLoc.GetFormattedText());
		_noFriendsIndicator.SetTextAutoSize(_friendsLoc.GetFormattedText());
	}

	public void Cleanup()
	{
		_loadCts?.Cancel();
		_loadCts?.Dispose();
		_leftArrow.Visible = false;
		_rightArrow.Visible = false;
		_loadingIndicator.Visible = false;
		_noScoresIndicator.Visible = false;
		_noFriendsIndicator.Visible = false;
		_paginator.Visible = false;
		if (_noScoreUploadIndicator != null)
		{
			_noScoreUploadIndicator.Visible = false;
		}
		ClearEntries();
	}

	public void Initialize(DateTimeOffset dateTime, IEnumerable<ulong> playersInRun, bool allowPagination)
	{
		_playersInRun.Clear();
		_playersInRun.AddRange(playersInRun);
		_paginator.Initialize(this, dateTime, allowPagination);
		_paginator.Visible = true;
		_todaysDailyTime = dateTime;
		SetDay(dateTime);
	}

	public void SetDay(DateTimeOffset dateTime)
	{
		_leaderboardTime = dateTime;
		SetPage(0);
	}

	private void ChangePage(int increment)
	{
		SetPage(_currentPage + increment);
	}

	private void SetPage(int page)
	{
		_currentPage = page;
		TaskHelper.RunSafely(LoadLeaderboard(_leaderboardTime, _currentPage));
	}

	private async Task LoadLeaderboard(DateTimeOffset dateTime, int page)
	{
		_loadCts?.Dispose();
		_loadCts = new CancellationTokenSource();
		CancellationToken ct = _loadCts.Token;
		ClearEntries();
		_rightArrow.Disable();
		_leftArrow.Disable();
		_paginator.Disable();
		_noFriendsIndicator.Visible = false;
		_noScoresIndicator.Visible = false;
		_loadingIndicator.Visible = true;
		try
		{
			string leaderboardName = DailyRunUtility.GetLeaderboardName(dateTime, _playersInRun.Count);
			DateTimeOffset dateTime2 = dateTime - TimeSpan.FromDays(1);
			DateTimeOffset rightLeaderboardTime = dateTime + TimeSpan.FromDays(1);
			Task<ILeaderboardHandle?> mainTask = LeaderboardManager.GetLeaderboard(leaderboardName);
			Task<ILeaderboardHandle?> leftTask = LeaderboardManager.GetLeaderboard(DailyRunUtility.GetLeaderboardName(dateTime2, _playersInRun.Count));
			Task<ILeaderboardHandle?> rightTask = LeaderboardManager.GetLeaderboard(DailyRunUtility.GetLeaderboardName(rightLeaderboardTime, _playersInRun.Count));
			global::_003C_003Ey__InlineArray3<Task<ILeaderboardHandle>> buffer = default(global::_003C_003Ey__InlineArray3<Task<ILeaderboardHandle>>);
			buffer[0] = mainTask;
			buffer[1] = leftTask;
			buffer[2] = rightTask;
			await Task.WhenAll<ILeaderboardHandle>(buffer).WaitAsync(ct);
			ILeaderboardHandle handle = await mainTask;
			if (handle != null)
			{
				List<LeaderboardEntry> list = await LeaderboardManager.QueryLeaderboard(handle, LeaderboardQueryType.Global, page * 10, 10).WaitAsync(ct);
				_noScoresIndicator.Visible = list.Count <= 0;
				FillEntries(list);
				_leftArrow.Visible = true;
				_rightArrow.Visible = true;
				if (page > 0)
				{
					_leftArrow.Enable();
				}
				else
				{
					_leftArrow.Disable();
				}
				if (page * 10 + 10 < LeaderboardManager.GetLeaderboardEntryCount(handle) && !_hasNegativeScore)
				{
					_rightArrow.Enable();
				}
				else
				{
					_rightArrow.Disable();
				}
			}
			else
			{
				_noScoresIndicator.Visible = true;
				_leftArrow.Visible = false;
				_rightArrow.Visible = false;
			}
			bool hasLeftLeaderboard = await leftTask != null;
			bool rightArrowEnabled = await rightTask != null || rightLeaderboardTime == _todaysDailyTime;
			_currentPage = page;
			_paginator.Enable(hasLeftLeaderboard, rightArrowEnabled);
			_loadingIndicator.Visible = false;
			if (_noScoreUploadIndicator != null && _todaysDailyTime == dateTime)
			{
				bool flag = await DailyRunUtility.ShouldUploadScore(handle, _playersInRun).WaitAsync(ct);
				_noScoreUploadIndicator.Visible = !flag;
			}
		}
		catch (OperationCanceledException)
		{
		}
	}

	private void FillEntries(List<LeaderboardEntry> entries)
	{
		if (entries.Count == 0)
		{
			return;
		}
		_hasNegativeScore = false;
		NDailyRunLeaderboardRow child = NDailyRunLeaderboardRow.CreateHeader();
		_scoreContainer.AddChildSafely(child);
		NDailyRunLeaderboardSeparator child2 = NDailyRunLeaderboardSeparator.Create();
		_scoreContainer.AddChildSafely(child2);
		foreach (LeaderboardEntry entry in entries)
		{
			if (entry.score < 0)
			{
				_hasNegativeScore = true;
				continue;
			}
			_scoreContainer.AddChildSafely(NDailyRunLeaderboardRow.Create(entry));
			_scoreContainer.AddChildSafely(NDailyRunLeaderboardSeparator.Create());
		}
	}

	private void ClearEntries()
	{
		_noScoresIndicator.Visible = false;
		_noFriendsIndicator.Visible = false;
		foreach (Node child in _scoreContainer.GetChildren())
		{
			child.QueueFreeSafely();
		}
	}
}
