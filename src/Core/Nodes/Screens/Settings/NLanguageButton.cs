using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NLanguageButton : NButton
{
	private TextureRect _image;

	private Control _outline;

	private TextureRect _flag;

	private MegaLabel _label;

	private Tween? _tween;

	public string isoCode;

	public bool IsSelected { get; private set; }

	public override void _Ready()
	{
		ConnectSignals();
		_image = GetNode<TextureRect>("%ButtonImage");
		_outline = GetNode<Control>("%Outline");
		_flag = GetNode<TextureRect>("%FlagImage");
		_label = GetNode<MegaLabel>("%Label");
	}

	public void Init(string languageIsoCode)
	{
		Log.Info("Adding Language: " + languageIsoCode);
		isoCode = languageIsoCode;
		_label.SetTextAutoSize(new LocString("settings_ui", "LANGUAGE_" + languageIsoCode.ToUpperInvariant()).GetRawText());
		_flag.Texture = ResourceLoader.Load<Texture2D>("res://images/ui/settings_screen/flag_" + languageIsoCode + ".png", null, ResourceLoader.CacheMode.Reuse);
		IsSelected = SaveManager.Instance.SettingsSave.Language == isoCode;
		_outline.Visible = IsSelected;
	}

	protected override void OnRelease()
	{
		base.OnRelease();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_image, "self_modulate", new Color("26373d"), 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(_image, "modulate", Colors.White, 0.05);
		_tween.TweenProperty(_image, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(_label, "modulate", StsColors.cream, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_image, "self_modulate", new Color("235161"), 0.03);
		_tween.TweenProperty(_image, "modulate", Colors.White, 0.05);
		_tween.TweenProperty(_image, "scale", Vector2.One * 1.05f, 0.03);
		_tween.TweenProperty(_label, "modulate", StsColors.gold, 0.03);
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_image, "modulate", Colors.White, 0.5);
		_tween.TweenProperty(_image, "self_modulate", new Color("26373d"), 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(_image, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(_label, "modulate", StsColors.cream, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	protected override void OnPress()
	{
		base.OnPress();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_image, "self_modulate", new Color("26373d"), 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(_image, "modulate", StsColors.lightGray, 0.05);
		_tween.TweenProperty(_image, "scale", Vector2.One * 0.95f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	public void SetAsSelected()
	{
		IsSelected = true;
		_outline.Visible = true;
	}

	public void SetAsDeselected()
	{
		IsSelected = false;
		_outline.Visible = false;
	}
}
