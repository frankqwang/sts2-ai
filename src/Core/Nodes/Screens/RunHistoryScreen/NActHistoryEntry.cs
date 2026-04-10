using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Runs.History;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens.RunHistoryScreen;

public partial class NActHistoryEntry : HBoxContainer
{
	private MegaLabel _actLabel;

	private LocString _actName;

	private RunHistory _runHistory;

	private IReadOnlyList<MapPointHistoryEntry> _entries;

	private int _baseFloorNum;

	private static string ScenePath => SceneHelper.GetScenePath("screens/run_history_screen/act_history_entry");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	public List<NMapPointHistoryEntry> Entries { get; private set; } = new List<NMapPointHistoryEntry>();

	public override void _Ready()
	{
		_actLabel = GetNode<MegaLabel>("%Title");
		_actLabel.SetTextAutoSize(_actName.GetFormattedText());
		for (int i = 0; i < _entries.Count; i++)
		{
			NMapPointHistoryEntry nMapPointHistoryEntry = NMapPointHistoryEntry.Create(_runHistory, _entries[i], i + _baseFloorNum);
			this.AddChildSafely(nMapPointHistoryEntry);
			Entries.Add(nMapPointHistoryEntry);
		}
	}

	public void SetPlayer(RunHistoryPlayer player)
	{
		foreach (NMapPointHistoryEntry entry in Entries)
		{
			entry.SetPlayer(player);
		}
	}

	public static NActHistoryEntry? Create(LocString actName, RunHistory runHistory, IReadOnlyList<MapPointHistoryEntry> logs, int baseFloorNum)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NActHistoryEntry nActHistoryEntry = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NActHistoryEntry>(PackedScene.GenEditState.Disabled);
		nActHistoryEntry._actName = actName;
		nActHistoryEntry._runHistory = runHistory;
		nActHistoryEntry._entries = logs;
		nActHistoryEntry._baseFloorNum = baseFloorNum;
		return nActHistoryEntry;
	}
}
