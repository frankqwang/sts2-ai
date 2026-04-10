using System;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public abstract partial class NSettingsSlider : Control
{
	protected NSlider _slider;

	private MegaLabel _valueLabel;

	private NSelectionReticle _selectionReticle;

	public override void _Ready()
	{
		if (GetType() != typeof(NSettingsSlider))
		{
			Log.Error($"{GetType()}");
			throw new InvalidOperationException("Don't call base._Ready()! Call ConnectSignals() instead.");
		}
		ConnectSignals();
	}

	protected virtual void ConnectSignals()
	{
		_slider = GetNode<NSlider>("Slider");
		_valueLabel = GetNode<MegaLabel>("SliderValue");
		_selectionReticle = GetNode<NSelectionReticle>("SelectionReticle");
		_slider.Connect(Godot.Range.SignalName.ValueChanged, Callable.From<double>(OnValueChanged));
		_valueLabel.SetTextAutoSize($"{_slider.Value}%");
		Connect(Control.SignalName.FocusEntered, Callable.From(OnFocus));
		Connect(Control.SignalName.FocusExited, Callable.From(OnUnfocus));
	}

	public override void _GuiInput(InputEvent input)
	{
		base._GuiInput(input);
		if (input.IsActionPressed(MegaInput.left))
		{
			_slider.Value -= 5.0;
		}
		if (input.IsActionPressed(MegaInput.right))
		{
			_slider.Value += 5.0;
		}
	}

	private void OnValueChanged(double value)
	{
		_valueLabel.SetTextAutoSize($"{value}%");
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
