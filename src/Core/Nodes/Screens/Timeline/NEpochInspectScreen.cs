using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Achievements;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Timeline;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Timeline;

public partial class NEpochInspectScreen : NClickableControl, IScreenContext
{
	private static readonly LocString placeholderLoc = new LocString("timeline", "PLACEHOLDER_PORTRAIT");

	public static readonly string lockedImagePath = ImageHelper.GetImagePath("packed/timeline/epoch_slot_locked.png");

	private NButton _closeButton;

	private TextureRect _portrait;

	private TextureRect _portraitFlash;

	private TextureRect _mask;

	private NEpochChains _chains;

	private ShaderMaterial _portraitHsv;

	private MegaRichTextLabel _fancyText;

	private MegaLabel _storyLabel;

	private MegaLabel _chapterLabel;

	private MegaLabel _closeLabel;

	private MegaLabel _placeholderLabel;

	private NEpochPaginateButton _nextChapterButton;

	private NEpochPaginateButton _prevChapterButton;

	private NUnlockInfo _unlockInfo;

	private List<SerializableEpoch> _allEpochs;

	private EpochModel _epoch;

	private EpochModel? _prevChapterEpoch;

	private EpochModel? _nextChapterEpoch;

	private LocString _chapterLoc;

	private bool _hasStory;

	private bool _wasRevealed;

	private float _prevChapterButtonOffsetX;

	private float _nextChapterButtonOffsetX;

	private float _maskOffsetX;

	private float _maskOffsetY;

	private float _closeButtonY;

	private Tween? _unlockTween;

	private Tween? _buttonTween;

	private Tween? _tween;

	private Tween? _textTween;

	private static readonly StringName _v = new StringName("v");

	private static readonly StringName _s = new StringName("s");

	public Control? DefaultFocusedControl => null;

	public override void _Ready()
	{
		_storyLabel = GetNode<MegaLabel>("%StoryLabel");
		_chapterLabel = GetNode<MegaLabel>("%ChapterLabel");
		_placeholderLabel = GetNode<MegaLabel>("%PlaceholderLabel");
		_portrait = GetNode<TextureRect>("%Portrait");
		_portraitFlash = GetNode<TextureRect>("%PortraitFlash");
		_mask = GetNode<TextureRect>("%Mask");
		_maskOffsetY = _mask.OffsetTop;
		_maskOffsetX = _mask.OffsetLeft;
		_chains = GetNode<NEpochChains>("%Chains");
		_portraitHsv = (ShaderMaterial)_portrait.Material;
		_closeButton = GetNode<NButton>("%CloseButton");
		_fancyText = GetNode<MegaRichTextLabel>("%FancyText");
		_closeLabel = GetNode<MegaLabel>("%CloseLabel");
		_chapterLoc = new LocString("timeline", "EPOCH_INSPECT.chapterFormat");
		_unlockInfo = GetNode<NUnlockInfo>("%UnlockInfo");
		_nextChapterButton = GetNode<NEpochPaginateButton>("%NextChapterButton");
		_prevChapterButton = GetNode<NEpochPaginateButton>("%PrevChapterButton");
		_prevChapterButtonOffsetX = _prevChapterButton.OffsetLeft;
		_nextChapterButtonOffsetX = _nextChapterButton.OffsetLeft;
		_nextChapterButton.Connect(NClickableControl.SignalName.MouseReleased, Callable.From<InputEvent>(delegate
		{
			NextChapter();
		}));
		_prevChapterButton.Connect(NClickableControl.SignalName.MouseReleased, Callable.From<InputEvent>(delegate
		{
			PrevChapter();
		}));
		Connect(NClickableControl.SignalName.MouseReleased, Callable.From<InputEvent>(OnMouseReleased));
		_closeButton.Disable();
	}

	public async Task Open(NEpochSlot slot, EpochModel epoch, bool wasRevealed)
	{
		_buttonTween?.FastForwardToCompletion();
		_wasRevealed = wasRevealed;
		_epoch = epoch;
		if (_epoch.IsArtPlaceholder)
		{
			_placeholderLabel.Visible = true;
			_placeholderLabel.Text = placeholderLoc.GetRawText();
		}
		else
		{
			_placeholderLabel.Visible = false;
		}
		base.Modulate = Colors.White;
		_portrait.Texture = epoch.BigPortrait;
		_fancyText.Modulate = StsColors.transparentWhite;
		_fancyText.Text = epoch.Description;
		_hasStory = epoch.StoryTitle != null;
		SfxCmd.Play("event:/sfx/ui/timeline/ui_timeline_open_epoch");
		if (_hasStory)
		{
			_storyLabel.SetTextAutoSize(epoch.StoryTitle ?? string.Empty);
			_chapterLoc.Add("ChapterIndex", epoch.ChapterIndex);
			_chapterLoc.Add("ChapterName", epoch.Title);
			_chapterLabel.SetTextAutoSize(_chapterLoc.GetFormattedText());
			_chapterLabel.VerticalAlignment = VerticalAlignment.Center;
		}
		else
		{
			_storyLabel.SetTextAutoSize(string.Empty);
			_chapterLabel.SetTextAutoSize(epoch.Title.GetFormattedText());
			_chapterLabel.VerticalAlignment = VerticalAlignment.Bottom;
			_nextChapterButton.Disable();
			_prevChapterButton.Disable();
		}
		_closeButton.MouseFilter = MouseFilterEnum.Stop;
		_closeButton.Scale = Vector2.One;
		_closeButtonY = _closeButton.Position.Y + 180f;
		_closeButton.Position = new Vector2(_closeButton.Position.X, _closeButtonY);
		_closeButton.Modulate = StsColors.transparentWhite;
		_closeButton.Disable();
		NTimelineScreen.Instance.ShowBackstopAndHideUi();
		base.Visible = true;
		Vector2 size = _mask.Size;
		_mask.GlobalPosition = slot.GlobalPosition;
		_mask.SetDeferred(Control.PropertyName.Size, slot.Size * slot.GetGlobalTransform().Scale);
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_mask, "offset_left", _maskOffsetX, 0.4).SetTrans(Tween.TransitionType.Cubic).SetEase(Tween.EaseType.Out);
		_tween.TweenProperty(_mask, "offset_top", _maskOffsetY, 0.4).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(_mask, "size", size, 0.4).SetTrans(Tween.TransitionType.Cubic).SetEase(Tween.EaseType.Out);
		_storyLabel.Modulate = StsColors.transparentWhite;
		_chapterLabel.Modulate = StsColors.transparentWhite;
		_nextChapterButton.Modulate = StsColors.transparentWhite;
		_prevChapterButton.Modulate = StsColors.transparentWhite;
		_tween.TweenProperty(_storyLabel, "modulate:a", 1f, 0.5).SetDelay(0.4);
		_tween.TweenProperty(_chapterLabel, "modulate:a", 1f, 0.5).SetDelay(0.2);
		_tween.TweenProperty(_prevChapterButton, "offset_left", _prevChapterButtonOffsetX, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back)
			.From(_prevChapterButtonOffsetX + 100f)
			.SetDelay(0.25);
		_tween.TweenProperty(_prevChapterButton, "modulate:a", 1f, 0.25).SetDelay(0.25);
		_tween.TweenProperty(_nextChapterButton, "offset_left", _nextChapterButtonOffsetX, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back)
			.From(_nextChapterButtonOffsetX - 100f)
			.SetDelay(0.25);
		_tween.TweenProperty(_nextChapterButton, "modulate:a", 1f, 0.25).SetDelay(0.25);
		if (wasRevealed)
		{
			await TaskHelper.RunSafely(UnlockAnimation(epoch));
			return;
		}
		_closeLabel.SetTextAutoSize(new LocString("timeline", "EPOCH_INSPECT.closeButton").GetRawText());
		_fancyText.Text = epoch.Description;
		_textTween?.Kill();
		_textTween = CreateTween().SetParallel();
		_textTween.TweenProperty(_fancyText, "modulate:a", 1f, 0.5).SetDelay(0.1);
		_fancyText.VisibleRatio = 1f;
		_buttonTween?.Kill();
		_buttonTween = CreateTween().SetParallel();
		_buttonTween.TweenProperty(_closeButton, "modulate:a", 1f, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic)
			.SetDelay(0.1);
		_buttonTween.TweenProperty(_closeButton, "position:y", _closeButtonY - 180f, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back)
			.SetDelay(0.1);
		_buttonTween.TweenCallback(Callable.From(_closeButton.Enable));
		RefreshChapterPaginators();
		_unlockInfo.AnimIn(epoch.UnlockText);
	}

	private void HidePaginators()
	{
		_hasStory = false;
		_nextChapterButton.Visible = false;
		_prevChapterButton.Visible = false;
	}

	private void OpenViaPaginator(EpochModel epoch)
	{
		_epoch = epoch;
		base.Modulate = Colors.White;
		_fancyText.Text = epoch.Description;
		_portrait.Texture = epoch.BigPortrait;
		_hasStory = epoch.StoryTitle != null;
		_storyLabel.Modulate = Colors.White;
		_chapterLabel.Modulate = Colors.White;
		if (_hasStory)
		{
			_storyLabel.SetTextAutoSize(epoch.StoryTitle ?? string.Empty);
			_chapterLoc.Add("ChapterIndex", epoch.ChapterIndex);
			_chapterLoc.Add("ChapterName", epoch.Title);
			_chapterLabel.SetTextAutoSize(_chapterLoc.GetFormattedText());
			_chapterLabel.VerticalAlignment = VerticalAlignment.Center;
			_nextChapterButton.Modulate = Colors.White;
			_prevChapterButton.Modulate = Colors.White;
		}
		else
		{
			_storyLabel.SetTextAutoSize(string.Empty);
			_chapterLabel.SetTextAutoSize(epoch.Title.GetFormattedText());
			_chapterLabel.VerticalAlignment = VerticalAlignment.Bottom;
			_nextChapterButton.Visible = false;
			_prevChapterButton.Visible = false;
		}
		_fancyText.Modulate = StsColors.transparentWhite;
		NTimelineScreen.Instance.ShowBackstopAndHideUi();
		base.Visible = true;
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_textTween?.Kill();
		_textTween = CreateTween().SetParallel();
		_tween.TweenProperty(_mask, "offset_top", _maskOffsetY, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_textTween.TweenProperty(_fancyText, "modulate:a", 1f, 0.5).SetDelay(0.1);
		_fancyText.VisibleRatio = 1f;
		TaskHelper.RunSafely(_unlockInfo.AnimInViaPaginator(epoch.UnlockText));
	}

	public void Close()
	{
		if (NTimelineScreen.Instance.IsScreenQueued())
		{
			NTimelineScreen.Instance.OpenQueuedScreen();
		}
		else
		{
			NTimelineScreen.Instance.EnableInput();
			NTimelineScreen.Instance.HideBackstopAndShowUi(showBackButton: true);
		}
		base.FocusMode = FocusModeEnum.None;
		_buttonTween?.FastForwardToCompletion();
		_unlockTween?.FastForwardToCompletion();
		_tween?.FastForwardToCompletion();
		_textTween?.FastForwardToCompletion();
		_buttonTween = CreateTween().SetParallel();
		_buttonTween.TweenProperty(_closeButton, "scale", new Vector2(3f, 0.1f), 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_buttonTween.TweenProperty(_closeButton, "modulate:a", 0f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		_buttonTween.TweenProperty(this, "modulate", new Color(0f, 0f, 0f, 0f), 0.5);
		_buttonTween.TweenCallback(Callable.From(delegate
		{
			base.Visible = false;
			_closeButton.Disable();
			if (_wasRevealed)
			{
				AchievementsHelper.CheckTimelineComplete();
			}
		}));
		NHotkeyManager.Instance.RemoveHotkeyPressedBinding(MegaInput.right, NextChapter);
		NHotkeyManager.Instance.RemoveHotkeyPressedBinding(MegaInput.left, PrevChapter);
	}

	public async Task UnlockAnimation(EpochModel epoch)
	{
		HidePaginators();
		epoch.QueueUnlocks();
		SaveManager.Instance.SaveProgressFile();
		_unlockInfo.HideImmediately();
		_closeLabel.SetTextAutoSize(new LocString("timeline", "EPOCH_INSPECT.continueButton").GetRawText());
		_fancyText.VisibleRatio = 0f;
		_fancyText.Modulate = StsColors.transparentWhite;
		_portraitHsv.SetShaderParameter(_s, 0f);
		_portraitHsv.SetShaderParameter(_v, 0.75f);
		_chains.Texture = PreloadManager.Cache.GetTexture2D(lockedImagePath);
		_chains.Visible = true;
		_chains.Modulate = Colors.White;
		_chains.SelfModulate = Colors.White;
		_portraitFlash.Modulate = new Color(1f, 1f, 1f, 0f);
		_unlockTween?.Kill();
		_unlockTween = CreateTween().SetParallel();
		_unlockTween.TweenProperty(_chains, "scale", Vector2.One * 0.98f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
			.SetDelay(0.5);
		await ToSignal(_unlockTween, Tween.SignalName.Finished);
		_chains.Unlock();
		await ToSignal(_chains, NEpochChains.SignalName.OnAnimationFinished);
		_portraitFlash.Modulate = Colors.White;
		_unlockTween = CreateTween().SetParallel();
		_unlockTween.TweenProperty(_portraitFlash, "modulate:a", 0f, 0.5);
		_unlockTween.TweenMethod(Callable.From<float>(UpdateShaderS), _portraitHsv.GetShaderParameter(_s), 1f, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_unlockTween.TweenMethod(Callable.From<float>(UpdateShaderV), _portraitHsv.GetShaderParameter(_v), 1f, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_textTween?.Kill();
		_textTween = CreateTween().SetParallel();
		_textTween.TweenProperty(_fancyText, "modulate:a", 1f, 2.0).SetDelay(0.25);
		_textTween.TweenProperty(_fancyText, "visible_ratio", 1f, (double)_fancyText.GetTotalCharacterCount() * 0.015).SetDelay(0.5);
		_buttonTween?.Kill();
		_buttonTween = CreateTween().SetParallel();
		_buttonTween.TweenProperty(_closeButton, "modulate:a", 1f, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic)
			.SetDelay(1.0);
		_buttonTween.TweenProperty(_closeButton, "position:y", _closeButtonY - 180f, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back)
			.SetDelay(1.0);
		_buttonTween.TweenCallback(Callable.From(_closeButton.Enable));
		await ToSignal(_unlockTween, Tween.SignalName.Finished);
	}

	private void UpdateShaderS(float value)
	{
		_portraitHsv.SetShaderParameter(_s, value);
	}

	private void UpdateShaderV(float value)
	{
		_portraitHsv.SetShaderParameter(_v, value);
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (inputEvent.IsActionPressed(MegaInput.select) || inputEvent.IsActionPressed(MegaInput.accept))
		{
			SpeedUpTextAnimation();
		}
	}

	private void OnMouseReleased(InputEvent obj)
	{
		SpeedUpTextAnimation();
	}

	private void SpeedUpTextAnimation()
	{
		if (_textTween != null && _textTween.IsRunning())
		{
			_textTween.Kill();
			_fancyText.Modulate = Colors.White;
			_fancyText.VisibleRatio = 1f;
		}
	}

	private void NextChapter()
	{
		OpenViaPaginator(_nextChapterEpoch);
		RefreshChapterPaginators();
	}

	private void PrevChapter()
	{
		OpenViaPaginator(_prevChapterEpoch);
		RefreshChapterPaginators();
	}

	private void RefreshChapterPaginators()
	{
		if (_hasStory)
		{
			_nextChapterEpoch = StoryModel.NextChapter(_epoch);
			_prevChapterEpoch = StoryModel.PrevChapter(_epoch);
			if (_nextChapterEpoch != null)
			{
				NHotkeyManager.Instance.PushHotkeyPressedBinding(MegaInput.right, NextChapter);
				_nextChapterButton.Visible = true;
			}
			else
			{
				_nextChapterButton.Visible = false;
			}
			if (_prevChapterEpoch != null)
			{
				NHotkeyManager.Instance.PushHotkeyPressedBinding(MegaInput.left, PrevChapter);
				_prevChapterButton.Visible = true;
			}
			else
			{
				_prevChapterButton.Visible = false;
			}
		}
	}
}
