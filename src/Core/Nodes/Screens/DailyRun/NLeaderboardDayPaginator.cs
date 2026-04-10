using System;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;

namespace MegaCrit.Sts2.Core.Nodes.Screens.DailyRun;

public partial class NLeaderboardDayPaginator : Control
{
	protected MegaLabel _label;

	private MegaLabel _vfxLabel;

	private NLeaderboardPageArrow _leftArrow;

	private NLeaderboardPageArrow _rightArrow;

	private NSelectionReticle _selectionReticle;

	private Tween? _tween;

	private const double _animDuration = 0.25;

	private const float _animDistance = 90f;

	private DateTimeOffset _currentDay;

	private NDailyRunLeaderboard? _leaderboard;

	public override void _Ready()
	{
		_label = GetNode<MegaLabel>("LabelContainer/Mask/Label");
		_vfxLabel = GetNode<MegaLabel>("LabelContainer/Mask/VfxLabel");
		_selectionReticle = GetNode<NSelectionReticle>("SelectionReticle");
		_leftArrow = GetNode<NLeaderboardPageArrow>("LeftArrow");
		_rightArrow = GetNode<NLeaderboardPageArrow>("RightArrow");
		Connect(Control.SignalName.FocusEntered, Callable.From(OnFocus));
		Connect(Control.SignalName.FocusExited, Callable.From(OnUnfocus));
		_leftArrow.Connect(PageLeft);
		_rightArrow.Connect(PageRight);
	}

	public void Initialize(NDailyRunLeaderboard leaderboard, DateTimeOffset dateTime, bool showArrows)
	{
		_currentDay = dateTime;
		_leaderboard = leaderboard;
		OnDayChanged(changeLeaderboardDay: false);
		_leftArrow.Visible = showArrows;
		_rightArrow.Visible = showArrows;
	}

	public override void _GuiInput(InputEvent input)
	{
		base._GuiInput(input);
		if (input.IsActionPressed(MegaInput.left))
		{
			PageLeft();
		}
		if (input.IsActionPressed(MegaInput.right))
		{
			PageRight();
		}
	}

	private void PageLeft()
	{
		_currentDay -= TimeSpan.FromDays(1);
		DayChangeHelper(pagedLeft: true);
	}

	private void PageRight()
	{
		_currentDay += TimeSpan.FromDays(1);
		DayChangeHelper(pagedLeft: false);
	}

	private void DayChangeHelper(bool pagedLeft)
	{
		_vfxLabel.SetTextAutoSize(_label.Text);
		_vfxLabel.Modulate = _label.Modulate;
		OnDayChanged(changeLeaderboardDay: true);
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_label, "position:x", 0f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic)
			.From(pagedLeft ? (-90f) : 90f);
		_tween.TweenProperty(_label, "modulate:a", 1f, 0.25).From(0.75f);
		_tween.TweenProperty(_vfxLabel, "position:x", pagedLeft ? 90f : (-90f), 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic)
			.From(0f);
		_tween.TweenProperty(_vfxLabel, "modulate", StsColors.transparentBlack, 0.25);
	}

	private void OnDayChanged(bool changeLeaderboardDay)
	{
		_label.SetTextAutoSize(_currentDay.ToString(NDailyRunScreen.dateFormat));
		if (changeLeaderboardDay)
		{
			_leaderboard.SetDay(_currentDay);
		}
	}

	public void Disable()
	{
		_leftArrow.Disable();
		_rightArrow.Disable();
	}

	public void Enable(bool leftArrowEnabled, bool rightArrowEnabled)
	{
		if (leftArrowEnabled)
		{
			_leftArrow.Enable();
		}
		if (rightArrowEnabled)
		{
			_rightArrow.Enable();
		}
	}

	private void OnFocus()
	{
		if (NControllerManager.Instance.IsUsingController)
		{
			_selectionReticle.OnSelect();
		}
	}

	private void OnUnfocus()
	{
		_selectionReticle.OnDeselect();
	}
}
