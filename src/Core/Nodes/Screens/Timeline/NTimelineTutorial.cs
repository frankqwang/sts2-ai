using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Timeline;

public partial class NTimelineTutorial : Control, IScreenContext
{
	private MegaRichTextLabel _text;

	private NAcknowledgeButton _acknowledgeButton;

	private NTimelineScreen _timeline;

	private Tween? _tween;

	public Control? DefaultFocusedControl => null;

	public void Init(NTimelineScreen screen)
	{
		_timeline = screen;
		screen.HideBackButtonImmediately();
	}

	public override void _Ready()
	{
		SfxCmd.Play("event:/sfx/ui/timeline/ui_timeline_unlock");
		_text = GetNode<MegaRichTextLabel>("%TutorialText");
		_text.Text = "[center]" + new LocString("timeline", "TUTORIAL_TEXT").GetRawText() + "[/center]";
		_acknowledgeButton = GetNode<NAcknowledgeButton>("%AcknowledgeButton");
		_acknowledgeButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(CloseTutorial));
		AnimateTutorial();
	}

	private void CloseTutorial(NButton _)
	{
		_acknowledgeButton.Disable();
		_tween?.FastForwardToCompletion();
		_tween = CreateTween();
		_tween.TweenProperty(this, "modulate:a", 0f, 0.5);
		_tween.Chain().TweenCallback(Callable.From(delegate
		{
			TaskHelper.RunSafely(_timeline.SpawnFirstTimeTimeline());
			this.QueueFreeSafely();
		}));
	}

	private void AnimateTutorial()
	{
		_acknowledgeButton.Disable();
		_text.VisibleRatio = 0f;
		MegaRichTextLabel text = _text;
		Color modulate = _text.Modulate;
		modulate.A = 0f;
		text.Modulate = modulate;
		_tween?.FastForwardToCompletion();
		_tween = CreateTween();
		_tween.TweenProperty(_text, "visible_ratio", 1f, 2.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Quad);
		_tween.Parallel().TweenProperty(_text, "modulate:a", 1f, 1.0);
		_tween.Chain().TweenCallback(Callable.From(delegate
		{
			_acknowledgeButton.Enable();
		}));
		_tween.Parallel().TweenProperty(_acknowledgeButton, "modulate:a", 1f, 0.3).SetEase(Tween.EaseType.Out)
			.SetTrans(Tween.TransitionType.Cubic)
			.SetDelay(1.0);
		_tween.Parallel().TweenProperty(_acknowledgeButton, "position:y", 920f, 0.3).SetEase(Tween.EaseType.Out)
			.SetTrans(Tween.TransitionType.Back)
			.SetDelay(1.0);
	}
}
