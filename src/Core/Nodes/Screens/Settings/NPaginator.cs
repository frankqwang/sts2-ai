using System;
using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NPaginator : Control
{
	protected MegaLabel _label;

	private MegaLabel _vfxLabel;

	private NSelectionReticle _selectionReticle;

	protected readonly List<string> _options = new List<string>();

	protected int _currentIndex;

	private Tween? _tween;

	private const double _animDuration = 0.25;

	private const float _animDistance = 90f;

	public override void _Ready()
	{
		if (GetType() != typeof(NPaginator))
		{
			throw new InvalidOperationException("Don't call base._Ready(). Use ConnectSignals() instead");
		}
		ConnectSignals();
	}

	protected void ConnectSignals()
	{
		_label = GetNode<MegaLabel>("%Label");
		_vfxLabel = GetNode<MegaLabel>("%VfxLabel");
		_selectionReticle = GetNode<NSelectionReticle>("SelectionReticle");
		Connect(Control.SignalName.FocusEntered, Callable.From(OnFocus));
		Connect(Control.SignalName.FocusExited, Callable.From(OnUnfocus));
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

	protected virtual void OnIndexChanged(int index)
	{
	}

	public void SetIndex(int index)
	{
		if (_currentIndex != index)
		{
			_currentIndex = Mathf.Clamp(index, 0, _options.Count - 1);
			OnIndexChanged(_currentIndex);
		}
	}

	public void PageLeft()
	{
		_currentIndex--;
		if (_currentIndex < 0)
		{
			_currentIndex = _options.Count - 1;
		}
		IndexChangeHelper(pagedLeft: true);
	}

	public void PageRight()
	{
		_currentIndex++;
		if (_currentIndex > _options.Count - 1)
		{
			_currentIndex = 0;
		}
		IndexChangeHelper(pagedLeft: false);
	}

	private void IndexChangeHelper(bool pagedLeft)
	{
		_vfxLabel.SetTextAutoSize(_label.Text);
		_vfxLabel.Modulate = _label.Modulate;
		OnIndexChanged(_currentIndex);
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_label, "position:x", 0f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic)
			.From(pagedLeft ? (-90f) : 90f);
		_tween.TweenProperty(_label, "modulate:a", 1f, 0.25).From(0.75f);
		_tween.TweenProperty(_vfxLabel, "position:x", pagedLeft ? 90f : (-90f), 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic)
			.From(0f);
		_tween.TweenProperty(_vfxLabel, "modulate", StsColors.transparentBlack, 0.25);
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
