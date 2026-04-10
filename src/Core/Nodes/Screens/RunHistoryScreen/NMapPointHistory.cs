using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Relics;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Screens.RunHistoryScreen;

public partial class NMapPointHistory : Control
{
	private Control _actContainer;

	private readonly List<NActHistoryEntry> _actHistories = new List<NActHistoryEntry>();

	private List<NMapPointHistoryEntry> MapHistories => _actHistories.SelectMany((NActHistoryEntry a) => a.Entries).ToList();

	public Control? DefaultFocusedControl => MapHistories.FirstOrDefault();

	public override void _Ready()
	{
		_actContainer = GetNode<Control>("%Acts");
	}

	public void LoadHistory(RunHistory history)
	{
		foreach (Node child in _actContainer.GetChildren())
		{
			child.QueueFreeSafely();
		}
		_actHistories.Clear();
		int num = 1;
		for (int i = 0; i < history.MapPointHistory.Count; i++)
		{
			LocString title = ModelDb.GetById<ActModel>(history.Acts[i]).Title;
			NActHistoryEntry nActHistoryEntry = NActHistoryEntry.Create(title, history, history.MapPointHistory[i], num);
			_actContainer.AddChildSafely(nActHistoryEntry);
			_actHistories.Add(nActHistoryEntry);
			num += history.MapPointHistory[i].Count;
		}
	}

	public void SetPlayer(RunHistoryPlayer player)
	{
		foreach (NActHistoryEntry actHistory in _actHistories)
		{
			actHistory.SetPlayer(player);
		}
	}

	public void SetDeckHistory(NDeckHistory deckHistory)
	{
		deckHistory.Connect(NDeckHistory.SignalName.Hovered, Callable.From<NDeckHistoryEntry>(HighlightRelevantEntries));
		deckHistory.Connect(NDeckHistory.SignalName.Unhovered, Callable.From((Action<NDeckHistoryEntry>)UnHighlightEntries));
	}

	public void SetRelicHistory(NRelicHistory relicHistory)
	{
		relicHistory.Connect(NRelicHistory.SignalName.Hovered, Callable.From<NRelicBasicHolder>(HighlightRelevantEntries));
		relicHistory.Connect(NRelicHistory.SignalName.Unhovered, Callable.From((Action<NRelicBasicHolder>)UnHighlightEntries));
	}

	private void HighlightRelevantEntries(NDeckHistoryEntry historyEntry)
	{
		foreach (int floorNumber in historyEntry.FloorsAddedToDeck)
		{
			MapHistories.FirstOrDefault((NMapPointHistoryEntry e) => e.FloorNum == floorNumber)?.Highlight();
		}
	}

	private void HighlightRelevantEntries(NRelicBasicHolder holder)
	{
		if (holder.Relic.Model.FloorAddedToDeck > 0)
		{
			MapHistories.FirstOrDefault((NMapPointHistoryEntry e) => e.FloorNum == holder.Relic.Model.FloorAddedToDeck)?.Highlight();
		}
	}

	private void UnHighlightEntries(NRelicBasicHolder _)
	{
		UnHighlightEntries();
	}

	private void UnHighlightEntries(NDeckHistoryEntry _)
	{
		UnHighlightEntries();
	}

	private void UnHighlightEntries()
	{
		foreach (NMapPointHistoryEntry mapHistory in MapHistories)
		{
			mapHistory.Unhighlight();
		}
	}
}
