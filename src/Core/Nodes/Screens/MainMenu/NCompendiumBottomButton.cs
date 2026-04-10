using System;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Localization.Fonts;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;

public partial class NCompendiumBottomButton : NButton
{
	private Control _bgPanel;

	private MegaLabel _label;

	private TextureRect _icon;

	private string? _locKeyPrefix;

	private ShaderMaterial _hsv;

	private static readonly StringName _v = new StringName("v");

	private float _defaultV;

	private float _focusV;

	private float _pressV;

	private Tween? _tween;

	public override void _Ready()
	{
		if (GetType() != typeof(NCompendiumBottomButton))
		{
			throw new InvalidOperationException("Don't call base._Ready(). Use ConnectSignals() instead.");
		}
		ConnectSignals();
	}

	protected override void ConnectSignals()
	{
		base.ConnectSignals();
		_bgPanel = GetNode<Control>("BgPanel");
		_hsv = (ShaderMaterial)_bgPanel.Material;
		_icon = GetNode<TextureRect>("Icon");
		_label = GetNode<MegaLabel>("Label");
		_defaultV = (float)_hsv.GetShaderParameter(_v);
		_focusV = _defaultV + 0.2f;
		_pressV = _defaultV - 0.2f;
	}

	public void SetLocalization(string locKeyPrefix)
	{
		_locKeyPrefix = locKeyPrefix;
		RefreshLabels();
	}

	public override void _Notification(int what)
	{
		if ((long)what == 2010 && _locKeyPrefix != null && IsNodeReady())
		{
			RefreshLabels();
		}
	}

	private void RefreshLabels()
	{
		LocString locString = new LocString("main_menu_ui", _locKeyPrefix + ".title");
		_label.SetTextAutoSize(locString.GetFormattedText());
		_label.ApplyLocaleFontSubstitution(FontType.Regular, ThemeConstants.Label.font);
	}

	protected override void OnEnable()
	{
	}

	protected override void OnDisable()
	{
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "scale", Vector2.One * 1.05f, 0.05);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderParam), _hsv.GetShaderParameter(_v), _focusV, 0.05);
	}

	protected override void OnUnfocus()
	{
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "scale", Vector2.One, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderParam), _hsv.GetShaderParameter(_v), _defaultV, 0.3);
		_tween.TweenProperty(this, "modulate", Colors.White, 0.3);
	}

	protected override void OnPress()
	{
		base.OnPress();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "scale", Vector2.One * 0.95f, 0.2).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		_tween.TweenProperty(this, "modulate", StsColors.lightGray, 0.2);
	}

	private void UpdateShaderParam(float newV)
	{
		_hsv.SetShaderParameter(_v, newV);
	}
}
