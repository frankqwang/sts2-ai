using System;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Events;

public partial class NDivinationButton : NButton
{
	private static readonly StringName _v = new StringName("v");

	private static readonly StringName _s = new StringName("s");

	private ShaderMaterial _hsv;

	private Tween? _tween;

	private Tween? _outlineTween;

	private MegaLabel _label;

	private Control _outline;

	private string[] _customHotkeys = Array.Empty<string>();

	protected override string[] Hotkeys => _customHotkeys;

	public override void _Ready()
	{
		ConnectSignals();
		_hsv = (ShaderMaterial)GetNode<TextureRect>("%ButtonImage").Material;
		_outline = GetNode<Control>("%Outline");
		_label = GetNode<MegaLabel>("%Label");
	}

	public void SetLabel(LocString loc)
	{
		_label.SetTextAutoSize(loc.GetFormattedText());
	}

	protected override void OnFocus()
	{
		if (!_outline.Visible)
		{
			base.OnFocus();
			_tween?.Kill();
			_tween = CreateTween().SetParallel();
			_tween.TweenMethod(Callable.From<float>(UpdateShaderS), _hsv.GetShaderParameter(_s), 1f, 0.05).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			_tween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 1f, 0.05).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		}
	}

	protected override void OnUnfocus()
	{
		if (!_outline.Visible)
		{
			base.OnUnfocus();
			_tween?.Kill();
			_tween = CreateTween().SetParallel();
			_tween.TweenMethod(Callable.From<float>(UpdateShaderS), _hsv.GetShaderParameter(_s), 0.9f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			_tween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 0.8f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		}
	}

	private void UpdateShaderS(float value)
	{
		_hsv.SetShaderParameter(_s, value);
	}

	private void UpdateShaderV(float value)
	{
		_hsv.SetShaderParameter(_v, value);
	}

	public void SetActive(bool isActive)
	{
		_outline.Visible = isActive;
	}

	public void SetHotkeys(string[] hotkeys)
	{
		UnregisterHotkeys();
		_customHotkeys = hotkeys;
		RegisterHotkeys();
		UpdateControllerButton();
	}
}
