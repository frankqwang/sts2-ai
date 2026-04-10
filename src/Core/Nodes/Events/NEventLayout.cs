using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Debug;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Vfx;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;

namespace MegaCrit.Sts2.Core.Nodes.Events;

public partial class NEventLayout : Control
{
	public const string defaultScenePath = "res://scenes/events/default_event_layout.tscn";

	protected Tween? _descriptionTween;

	protected VBoxContainer _optionsContainer;

	private TextureRect? _portrait;

	private MegaLabel? _title;

	protected EventModel _event;

	protected MegaLabel? _sharedEventLabel;

	private static readonly LocString _sharedEventLoc = new LocString("events", "SHARED_EVENT_INFO");

	protected MegaRichTextLabel? _description;

	private static bool _isDebugUiVisible;

	public Control? VfxContainer { get; private set; }

	public IEnumerable<NEventOptionButton> OptionButtons => _optionsContainer.GetChildren().OfType<NEventOptionButton>();

	public virtual Control? DefaultFocusedControl => OptionButtons.FirstOrDefault();

	public override void _Ready()
	{
		_portrait = GetNodeOrNull<TextureRect>("%Portrait");
		_title = GetNodeOrNull<MegaLabel>("%Title");
		_description = GetNodeOrNull<MegaRichTextLabel>("%EventDescription");
		VfxContainer = GetNodeOrNull<Control>("%VfxContainer");
		_sharedEventLabel = GetNodeOrNull<MegaLabel>("%SharedEventLabel");
		_sharedEventLabel?.SetTextAutoSize(_sharedEventLoc.GetFormattedText());
		_optionsContainer = GetNode<VBoxContainer>("%OptionsContainer");
		_description?.SetText(string.Empty);
		ApplyDebugUiVisibility();
	}

	public override void _EnterTree()
	{
		RunManager.Instance.EventSynchronizer.PlayerVoteChanged += OnPlayerVoteChanged;
	}

	public override void _ExitTree()
	{
		RunManager.Instance.EventSynchronizer.PlayerVoteChanged -= OnPlayerVoteChanged;
	}

	public virtual void SetEvent(EventModel eventModel)
	{
		_event = eventModel;
		InitializeVisuals();
		_event.OnRoomEnter();
	}

	protected virtual void InitializeVisuals()
	{
		SetPortrait(_event.CreateInitialPortrait());
		if (_event.HasVfx)
		{
			Node2D node2D = _event.CreateVfx();
			NEventRoom.Instance.Layout.AddVfxAnchoredToPortrait(node2D);
			node2D.Position = EventModel.VfxOffset;
		}
	}

	public void SetPortrait(Texture2D portrait)
	{
		if (_portrait == null)
		{
			throw new InvalidOperationException("Trying to set a portrait in an event layout that doesn't have one.");
		}
		_portrait.Texture = portrait;
	}

	public void AddVfxAnchoredToPortrait(Node? vfx)
	{
		_portrait.AddChildSafely(vfx);
	}

	public void RemoveNodesOnPortrait()
	{
		foreach (Node child in _portrait.GetChildren())
		{
			_portrait.RemoveChildSafely(child);
		}
	}

	public void SetTitle(string title)
	{
		if (_title != null)
		{
			_title.Text = title;
		}
	}

	public void SetDescription(string description)
	{
		if (_description != null)
		{
			_description.SetTextAutoSize(description);
			AnimateIn();
		}
	}

	protected virtual void AnimateIn()
	{
		if (_sharedEventLabel != null)
		{
			_sharedEventLabel.Modulate = StsColors.transparentWhite;
		}
		if (_description != null)
		{
			_description.Modulate = StsColors.transparentWhite;
			bool flag = SaveManager.Instance.PrefsSave.FastMode == FastModeType.Fast;
			_descriptionTween?.Kill();
			_descriptionTween = CreateTween().SetParallel();
			_descriptionTween.TweenInterval(flag ? 0.2 : 0.5);
			_descriptionTween.Chain();
			if (_title != null)
			{
				_descriptionTween.TweenProperty(_title, "modulate", Colors.White, flag ? 0.25 : 0.5);
			}
			_descriptionTween.TweenProperty(_description, "modulate", Colors.White, flag ? 0.5 : 1.0).SetDelay(0.25);
			_descriptionTween.TweenProperty(_description, "visible_ratio", 1f, flag ? 0.5 : 1.0).SetDelay(0.25).From(0f)
				.SetEase(Tween.EaseType.Out)
				.SetTrans(Tween.TransitionType.Sine);
			if (_sharedEventLabel != null)
			{
				_descriptionTween.TweenProperty(_sharedEventLabel, "modulate", Colors.White, flag ? 0.25 : 0.5).SetDelay(0.25);
			}
		}
	}

	public void ClearOptions()
	{
		foreach (Node item in _optionsContainer.GetChildren().ToList())
		{
			_optionsContainer.RemoveChildSafely(item);
			item.QueueFreeSafely();
		}
	}

	public void AddOptions(IEnumerable<EventOption> options)
	{
		if (_sharedEventLabel != null)
		{
			MegaLabel? sharedEventLabel = _sharedEventLabel;
			EventModel eventModel = _event;
			sharedEventLabel.Visible = eventModel != null && eventModel.IsShared && !eventModel.IsFinished && _event.Owner.RunState.Players.Count > 1;
		}
		foreach (EventOption option in options)
		{
			NEventOptionButton nEventOptionButton = NEventOptionButton.Create(_event, option, _optionsContainer.GetChildCount());
			_optionsContainer.AddChildSafely(nEventOptionButton);
			nEventOptionButton.RefreshVotes();
		}
		int childCount = _optionsContainer.GetChildCount();
		if (childCount != 0)
		{
			NodePath path = _optionsContainer.GetChild<Control>(0).GetPath();
			NodePath path2 = _optionsContainer.GetChild<Control>(childCount - 1).GetPath();
			for (int i = 0; i < childCount; i++)
			{
				Control child = _optionsContainer.GetChild<Control>(i);
				NodePath focusNeighborRight = (child.FocusNeighborLeft = child.GetPath());
				child.FocusNeighborRight = focusNeighborRight;
				child.FocusNeighborTop = ((i > 0) ? _optionsContainer.GetChild<Control>(i - 1).GetPath() : path2);
				child.FocusNeighborBottom = ((i < childCount - 1) ? _optionsContainer.GetChild<Control>(i + 1).GetPath() : path);
			}
			AnimateButtonsIn();
		}
	}

	public virtual void OnSetupComplete()
	{
	}

	protected virtual void AnimateButtonsIn()
	{
		foreach (NEventOptionButton button in OptionButtons)
		{
			Callable.From(delegate
			{
				button.AnimateIn();
			}).CallDeferred();
		}
	}

	public async Task BeforeSharedOptionChosen(EventOption option)
	{
		NEventOptionButton chosenButton = null;
		foreach (NEventOptionButton optionButton in OptionButtons)
		{
			optionButton.Disable();
			if (optionButton.Option == option)
			{
				chosenButton = optionButton;
			}
		}
		if (chosenButton == null)
		{
			return;
		}
		EventSplitVoteAnimation eventSplitVoteAnimation = new EventSplitVoteAnimation(this, _event.Owner.RunState);
		await eventSplitVoteAnimation.TryPlay(chosenButton);
		foreach (NEventOptionButton optionButton2 in OptionButtons)
		{
			if (optionButton2.Option != option)
			{
				optionButton2.GrayOut();
			}
		}
		await chosenButton.FlashConfirmation();
	}

	private void OnPlayerVoteChanged(Player player)
	{
		foreach (NEventOptionButton optionButton in OptionButtons)
		{
			optionButton.RefreshVotes();
		}
	}

	public void DisableEventOptions()
	{
		foreach (NEventOptionButton optionButton in OptionButtons)
		{
			optionButton.Disable();
		}
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (inputEvent.IsActionReleased(DebugHotkey.hideEventUi))
		{
			_isDebugUiVisible = !_isDebugUiVisible;
			ApplyDebugUiVisibility();
			NGame.Instance.AddChildSafely(NFullscreenTextVfx.Create(_isDebugUiVisible ? "Hide Event UI" : "Show Event UI"));
		}
	}

	private void ApplyDebugUiVisibility()
	{
		if (_isDebugUiVisible)
		{
			_optionsContainer.Visible = false;
			if (_title != null)
			{
				_title.Modulate = Colors.Transparent;
			}
			if (_description != null)
			{
				_description.Visible = false;
			}
		}
		else
		{
			_optionsContainer.Visible = true;
			if (_description != null)
			{
				_description.Visible = true;
			}
		}
	}
}
