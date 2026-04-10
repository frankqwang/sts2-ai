using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Ancients;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Rooms;

public partial class NEventRoom : Control, IScreenContext
{
	private EventModel _event;

	private IRunState _runState = NullRunState.Instance;

	private bool _isPreFinished;

	private NSceneContainer _eventContainer;

	private const string _scenePath = "res://scenes/rooms/event_room.tscn";

	private readonly List<EventOption> _connectedOptions = new List<EventOption>();

	public static NEventRoom? Instance => NRun.Instance?.EventRoom;

	public NEventLayout? Layout => _eventContainer.CurrentScene as NEventLayout;

	public ICustomEventNode? CustomEventNode => _eventContainer.CurrentScene as ICustomEventNode;

	public NCombatRoom? EmbeddedCombatRoom => (Layout as NCombatEventLayout)?.EmbeddedCombatRoom;

	public Control? VfxContainer { get; private set; }

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>("res://scenes/rooms/event_room.tscn");

	public Control? DefaultFocusedControl
	{
		get
		{
			IScreenContext customEventNode = CustomEventNode;
			if (customEventNode != null)
			{
				return customEventNode.DefaultFocusedControl;
			}
			return Layout?.DefaultFocusedControl;
		}
	}

	public static NEventRoom? Create(EventModel eventModel, IRunState? runState, bool isPreFinished)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NEventRoom nEventRoom = PreloadManager.Cache.GetScene("res://scenes/rooms/event_room.tscn").Instantiate<NEventRoom>(PackedScene.GenEditState.Disabled);
		nEventRoom._event = eventModel;
		nEventRoom._isPreFinished = isPreFinished;
		if (runState != null)
		{
			nEventRoom._runState = runState;
		}
		return nEventRoom;
	}

	public override void _Ready()
	{
		if (_event.Node != null)
		{
			throw new InvalidOperationException("Tried to create event room, but event already has a node!");
		}
		_eventContainer = GetNode<NSceneContainer>("%EventContainer");
		NGame.Instance.SetScreenShakeTarget(_eventContainer);
		Control control = _event.CreateScene().Instantiate<Control>(PackedScene.GenEditState.Disabled);
		_event.SetNode(control);
		_eventContainer.SetCurrentScene(control);
		VfxContainer = Layout?.VfxContainer;
		TaskHelper.RunSafely(SetupLayout());
	}

	public override void _ExitTree()
	{
		NGame.Instance.ClearScreenShakeTarget();
		_event.StateChanged -= RefreshEventState;
		_event.EnteringEventCombat -= OnEnteringEventCombat;
		foreach (EventOption connectedOption in _connectedOptions)
		{
			connectedOption.BeforeChosen -= BeforeOptionChosen;
		}
		_connectedOptions.Clear();
	}

	private async Task SetupLayout()
	{
		if (_event.Owner == null)
		{
			throw new InvalidOperationException("Event must be started before passed to NEventRoom!");
		}
		if (Layout == null)
		{
			return;
		}
		Layout.SetEvent(_event);
		SetTitle(_event.Title);
		_event.StateChanged += RefreshEventState;
		_event.EnteringEventCombat += OnEnteringEventCombat;
		await Cmd.Wait(0.2f);
		SetDescription(GetDescriptionOrFallback());
		if (_event is AncientEventModel ancientEventModel && !_isPreFinished)
		{
			ModelId id = _event.Owner.Character.Id;
			AncientStats statsForAncient = SaveManager.Instance.Progress.GetStatsForAncient(ancientEventModel.Id);
			int charVisits = statsForAncient?.GetVisitsAs(id) ?? 0;
			int totalVisits = statsForAncient?.TotalVisits ?? 0;
			IEnumerable<AncientDialogue> validDialogues = ancientEventModel.DialogueSet.GetValidDialogues(id, charVisits, totalVisits, !ancientEventModel.AnyCharacterDialogueBlacklist.Contains(_event.Owner.Character));
			AncientDialogue ancientDialogue = Rng.Chaotic.NextItem(validDialogues);
			foreach (AncientDialogueLine line in ancientDialogue.Lines)
			{
				line.LineText?.Add("Act1Name", _runState.Acts[0].Title);
			}
			((NAncientEventLayout)Layout).SetDialogue(ancientDialogue.Lines);
		}
		SetOptions(_event);
		Layout.OnSetupComplete();
	}

	public void SetPortrait(Texture2D portrait)
	{
		Layout.SetPortrait(portrait);
	}

	private void SetTitle(LocString title)
	{
		Layout.SetTitle(title.GetFormattedText());
	}

	private void SetDescription(LocString description)
	{
		if (description.Exists())
		{
			CharacterModel character = _event.Owner.Character;
			character.AddDetailsTo(description);
			description.Add("IsMultiplayer", _event.Owner.RunState.Players.Count > 1);
			_event.DynamicVars.AddTo(description);
			Layout.SetDescription(description.GetFormattedText());
		}
	}

	private void SetOptions(EventModel eventModel)
	{
		Layout.ClearOptions();
		IReadOnlyList<EventOption> readOnlyList = eventModel.CurrentOptions;
		if (eventModel.IsFinished)
		{
			readOnlyList = new global::_003C_003Ez__ReadOnlySingleElementList<EventOption>(new EventOption(eventModel, Proceed, "PROCEED", false, true));
		}
		foreach (EventOption item in readOnlyList)
		{
			item.BeforeChosen += BeforeOptionChosen;
			_connectedOptions.Add(item);
		}
		Layout.AddOptions(readOnlyList);
		ActiveScreenContext.Instance.Update();
	}

	public void OptionButtonClicked(EventOption option, int index)
	{
		if (option.IsLocked)
		{
			return;
		}
		if (option.IsProceed)
		{
			TaskHelper.RunSafely(option.Chosen());
			return;
		}
		if (!_event.IsShared)
		{
			Layout.ClearOptions();
		}
		RunManager.Instance.EventSynchronizer.ChooseLocalOption(index);
	}

	private async Task BeforeOptionChosen(EventOption option)
	{
		if (_event.Owner.RunState.Players.Count > 1 && RunManager.Instance.EventSynchronizer.IsShared && !option.IsProceed)
		{
			await Layout.BeforeSharedOptionChosen(option);
		}
		else if (!option.IsProceed)
		{
			DisableOptionButtons();
		}
	}

	private void RefreshEventState(EventModel eventModel)
	{
		SetDescription(GetDescriptionOrFallback());
		if (eventModel is AncientEventModel)
		{
			((NAncientEventLayout)Layout).ClearDialogue();
		}
		SetOptions(_event);
	}

	private void DisableOptionButtons()
	{
		Layout?.DisableEventOptions();
	}

	private void OnEnteringEventCombat()
	{
		DisableOptionButtons();
		if (Layout is NCombatEventLayout nCombatEventLayout)
		{
			nCombatEventLayout.HideEventVisuals();
		}
	}

	public static Task Proceed()
	{
		NMapScreen.Instance.SetTravelEnabled(enabled: true);
		NMapScreen.Instance.Open();
		return Task.CompletedTask;
	}

	private LocString GetDescriptionOrFallback()
	{
		return _event.Description ?? new LocString("events", "ERROR.description");
	}
}
