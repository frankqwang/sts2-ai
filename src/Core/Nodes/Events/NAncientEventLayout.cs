using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Ancients;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Nodes.Vfx;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;

namespace MegaCrit.Sts2.Core.Nodes.Events;

public partial class NAncientEventLayout : NEventLayout
{
	public const string ancientScenePath = "res://scenes/events/ancient_event_layout.tscn";

	private const double _contentTweenDuration = 1.0;

	private AncientEventModel _ancientEvent;

	private readonly List<AncientDialogueLine> _dialogue = new List<AncientDialogueLine>();

	private int _currentDialogueLine;

	private NAncientBgContainer _ancientBgContainer;

	private Control? _ancientNameBanner;

	private Tween? _bannerTween;

	private Control _contentContainer;

	private float _originalContentContainerHeight;

	private VBoxContainer _content;

	private VBoxContainer _dialogueContainer;

	private NAncientDialogueHitbox _dialogueHitbox;

	private Control _fakeNextButtonContainer;

	private Control _fakeNextButton;

	private TextureRect _fakeNextButtonControllerIcon;

	private MegaLabel _fakeNextButtonLabel;

	private Tween? _contentTween;

	private bool IsDialogueOnLastLine => _currentDialogueLine >= _dialogue.Count - 1;

	public override Control? DefaultFocusedControl
	{
		get
		{
			if (!IsDialogueOnLastLine)
			{
				return null;
			}
			return base.OptionButtons.FirstOrDefault();
		}
	}

	public override void _Ready()
	{
		base._Ready();
		_ancientBgContainer = GetNode<NAncientBgContainer>("%AncientBgContainer");
		_contentContainer = GetNode<Control>("%ContentContainer");
		_content = GetNode<VBoxContainer>("%Content");
		_dialogueContainer = GetNode<VBoxContainer>("%DialogueContainer");
		_dialogueHitbox = GetNode<NAncientDialogueHitbox>("%DialogueHitbox");
		_dialogueHitbox.Connect(NClickableControl.SignalName.Released, Callable.From<NClickableControl>(OnDialogueHitboxClicked));
		_dialogueHitbox.Visible = false;
		_dialogueHitbox.Disable();
		_fakeNextButtonContainer = GetNode<Control>("%FakeNextButtonContainer");
		_fakeNextButton = _fakeNextButtonContainer.GetNode<Control>("FakeNextButton");
		_fakeNextButtonLabel = _fakeNextButton.GetNode<MegaLabel>("Label");
		_fakeNextButtonControllerIcon = _fakeNextButton.GetNode<TextureRect>("ControllerIcon");
		_originalContentContainerHeight = _contentContainer.Size.Y;
		_contentContainer.Size = new Vector2(_contentContainer.Size.X, _fakeNextButtonContainer.GlobalPosition.Y - _contentContainer.GlobalPosition.Y);
		UpdateHotkeyDisplay();
		NControllerManager.Instance.Connect(NControllerManager.SignalName.MouseDetected, Callable.From(UpdateHotkeyDisplay));
		NControllerManager.Instance.Connect(NControllerManager.SignalName.ControllerDetected, Callable.From(UpdateHotkeyDisplay));
		NInputManager.Instance.Connect(NInputManager.SignalName.InputRebound, Callable.From(UpdateHotkeyDisplay));
	}

	public override void _EnterTree()
	{
		base._EnterTree();
		ActiveScreenContext.Instance.Updated += UpdateBannerVisibility;
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		if (_ancientEvent.HasAmbientBgm)
		{
			SfxCmd.StopLoop(_ancientEvent.AmbientBgm);
		}
		ActiveScreenContext.Instance.Updated -= UpdateBannerVisibility;
		NControllerManager.Instance.Disconnect(NControllerManager.SignalName.MouseDetected, Callable.From(UpdateHotkeyDisplay));
		NControllerManager.Instance.Disconnect(NControllerManager.SignalName.ControllerDetected, Callable.From(UpdateHotkeyDisplay));
		NInputManager.Instance.Disconnect(NInputManager.SignalName.InputRebound, Callable.From(UpdateHotkeyDisplay));
	}

	protected override void InitializeVisuals()
	{
		_ancientEvent = (AncientEventModel)_event;
		_ancientNameBanner = NAncientNameBanner.Create(_ancientEvent);
		this.AddChildSafely(_ancientNameBanner);
		UpdateBannerVisibility();
		AncientEventModel ancientEvent = _ancientEvent;
		if (ancientEvent != null && ancientEvent.Owner != null && ancientEvent.HealedAmount > 0)
		{
			TaskHelper.RunSafely(PlayHealVfxAfterFadeIn(_ancientEvent.Owner, _ancientEvent.HealedAmount));
		}
		foreach (Node child in _ancientBgContainer.GetChildren())
		{
			_ancientBgContainer.RemoveChildSafely(child);
		}
		_ancientBgContainer.AddChildSafely(_ancientEvent.CreateBackgroundScene().Instantiate<Control>(PackedScene.GenEditState.Disabled));
		if (_ancientEvent.HasAmbientBgm)
		{
			SfxCmd.PlayLoop(_ancientEvent.AmbientBgm);
		}
	}

	protected override void AnimateIn()
	{
		if (_description != null)
		{
			_description.Modulate = Colors.Transparent;
			_descriptionTween?.Kill();
			_descriptionTween = CreateTween().SetParallel();
			if (SaveManager.Instance.PrefsSave.FastMode == FastModeType.Fast)
			{
				_descriptionTween.TweenInterval(0.2);
			}
			else
			{
				_descriptionTween.TweenInterval(0.5);
			}
			_descriptionTween.Chain();
			_descriptionTween.TweenProperty(_description, "modulate", Colors.White, 1.0);
		}
	}

	public void SetDialogue(IReadOnlyList<AncientDialogueLine> lines)
	{
		_dialogue.Clear();
		_dialogue.AddRange(lines);
		_currentDialogueLine = 0;
		foreach (AncientDialogueLine line in lines)
		{
			NAncientDialogueLine child = NAncientDialogueLine.Create(line, _ancientEvent, _ancientEvent.Owner.Character);
			_dialogueContainer.AddChildSafely(child);
		}
	}

	public void ClearDialogue()
	{
		_dialogue.Clear();
		_dialogueContainer.FreeChildren();
	}

	public override void OnSetupComplete()
	{
		_dialogueContainer.ResetSize();
		_optionsContainer.ResetSize();
		_content.ResetSize();
		_content.Position = new Vector2(_content.Position.X, _contentContainer.Size.Y);
		SetDialogueLineAndAnimate(0);
	}

	protected override void AnimateButtonsIn()
	{
		foreach (NEventOptionButton optionButton in base.OptionButtons)
		{
			optionButton.Modulate = Colors.White;
			optionButton.EnableButton();
		}
	}

	private async Task PlayHealVfxAfterFadeIn(Player player, decimal healAmount)
	{
		await Cmd.Wait(0.2f);
		PlayerFullscreenHealVfx.Play(player, healAmount, base.VfxContainer);
	}

	private void OnDialogueHitboxClicked(NClickableControl _)
	{
		SetDialogueLineAndAnimate(_currentDialogueLine + 1);
	}

	private void SetDialogueLineAndAnimate(int lineIndex)
	{
		_currentDialogueLine = lineIndex;
		if (_contentTween != null)
		{
			_contentTween.Pause();
			_contentTween.CustomStep(1.0);
			_contentTween.Kill();
			_contentTween = null;
		}
		UpdateFakeNextButton();
		NAncientDialogueLine childOrNull = _dialogueContainer.GetChildOrNull<NAncientDialogueLine>(_currentDialogueLine);
		childOrNull?.PlaySfx();
		float num = ((childOrNull != null) ? (childOrNull.Position.Y + childOrNull.Size.Y) : 0f);
		if (IsDialogueOnLastLine)
		{
			_fakeNextButtonContainer.Visible = false;
			_contentContainer.Size = new Vector2(_contentContainer.Size.X, _originalContentContainerHeight);
			_dialogueHitbox.Visible = false;
			_dialogueHitbox.Disable();
			foreach (NEventOptionButton optionButton in base.OptionButtons)
			{
				optionButton.EnableButton();
			}
			num += _optionsContainer.Size.Y + 10f;
		}
		else
		{
			_fakeNextButtonContainer.Visible = true;
			_dialogueHitbox.Visible = true;
			_dialogueHitbox.Enable();
		}
		if (_dialogueContainer.GetChildCount() > _currentDialogueLine)
		{
			_dialogueContainer.GetChild<NAncientDialogueLine>(_currentDialogueLine).SetSpeakerIconVisible();
		}
		foreach (NEventOptionButton optionButton2 in base.OptionButtons)
		{
			optionButton2.FocusMode = FocusModeEnum.None;
		}
		_contentTween = CreateTween();
		_contentTween.TweenProperty(_content, "position", new Vector2(_content.Position.X, _contentContainer.Size.Y - num), 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		if (IsDialogueOnLastLine)
		{
			_contentTween.Parallel().TweenCallback(Callable.From(delegate
			{
				foreach (NEventOptionButton optionButton3 in base.OptionButtons)
				{
					optionButton3.FocusMode = FocusModeEnum.All;
				}
				DefaultFocusedControl?.TryGrabFocus();
			})).SetDelay(0.8);
		}
		for (int num2 = 0; num2 < _currentDialogueLine; num2++)
		{
			NAncientDialogueLine child = _dialogueContainer.GetChild<NAncientDialogueLine>(num2);
			child.SetTransparency((num2 != _currentDialogueLine) ? 0.25f : 1f);
		}
	}

	private void UpdateFakeNextButton()
	{
		LocString locString = ((_dialogue.Count > _currentDialogueLine) ? _dialogue[_currentDialogueLine].NextButtonText : null);
		if (locString != null)
		{
			_fakeNextButtonLabel.SetTextAutoSize(locString.GetFormattedText() ?? "");
		}
		else
		{
			_fakeNextButtonLabel.Visible = false;
		}
	}

	private void HideNameBanner()
	{
		if (_ancientNameBanner != null)
		{
			_bannerTween?.Kill();
			_bannerTween = CreateTween();
			_bannerTween.TweenProperty(_ancientNameBanner, "modulate:a", 0f, 0.25);
		}
	}

	private void ShowNameBanner()
	{
		if (_ancientNameBanner != null)
		{
			_bannerTween?.Kill();
			_bannerTween = CreateTween();
			_bannerTween.TweenProperty(_ancientNameBanner, "modulate:a", 1f, 0.5);
		}
	}

	private void UpdateBannerVisibility()
	{
		if (NEventRoom.Instance != null)
		{
			if (ActiveScreenContext.Instance.IsCurrent(NEventRoom.Instance))
			{
				ShowNameBanner();
			}
			else
			{
				HideNameBanner();
			}
		}
	}

	private void UpdateHotkeyDisplay()
	{
		_fakeNextButtonControllerIcon.Visible = NControllerManager.Instance.IsUsingController;
		string hotkey = _dialogueHitbox.GetHotkey();
		if (hotkey != null)
		{
			_fakeNextButtonControllerIcon.Texture = NInputManager.Instance.GetHotkeyIcon(hotkey);
		}
	}
}
