using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.CommonUi;

public partial class NPopupYesNoButton : NButton
{
	private static readonly StringName _v = new StringName("v");

	private static readonly StringName _s = new StringName("s");

	private Control _visuals;

	private Control _outline;

	private Control _image;

	private MegaLabel _label;

	private Tween? _tween;

	private float _baseS;

	private float _baseV;

	private ShaderMaterial _hsv;

	private CanvasItemMaterial _outlineMaterial;

	private bool _isFocused;

	private static readonly Color _goldOutline = new Color("f0b400");

	private bool _isYes;

	public bool IsYes
	{
		get
		{
			return _isYes;
		}
		set
		{
			DisconnectHotkeys();
			_isYes = value;
			Callable.From(base.RegisterHotkeys).CallDeferred();
			UpdateControllerButton();
		}
	}

	protected override string[] Hotkeys => new string[1] { _isYes ? MegaInput.select : MegaInput.cancel };

	public override void _Ready()
	{
		ConnectSignals();
		_visuals = GetNode<Control>("%Visuals");
		_outline = GetNode<Control>("%Outline");
		_image = GetNode<Control>("%Image");
		_label = GetNode<MegaLabel>("%Label");
		_hsv = (ShaderMaterial)_image.GetMaterial();
		_outlineMaterial = (CanvasItemMaterial)_outline.GetMaterial();
		_baseS = (float)_hsv.GetShaderParameter(_s);
		_baseV = (float)_hsv.GetShaderParameter(_v);
	}

	public override void _ExitTree()
	{
		DisconnectHotkeys();
	}

	public void DisconnectHotkeys()
	{
		string[] hotkeys = Hotkeys;
		foreach (string hotkey in hotkeys)
		{
			NHotkeyManager.Instance.RemoveHotkeyPressedBinding(hotkey, base.OnPressHandler);
			NHotkeyManager.Instance.RemoveHotkeyReleasedBinding(hotkey, base.OnReleaseHandler);
		}
	}

	public void SetText(string text)
	{
		_label.SetTextAutoSize(text);
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_isFocused = true;
		_outlineMaterial.BlendMode = CanvasItemMaterial.BlendModeEnum.Add;
		_outline.Modulate = Colors.White;
		_outline.SelfModulate = _goldOutline;
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_visuals, "scale", Vector2.One * 1.025f, 0.05);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderS), _hsv.GetShaderParameter(_s), _baseS + 0.25f, 0.05).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), _baseV + 0.25f, 0.05).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		_isFocused = false;
		_outlineMaterial.BlendMode = CanvasItemMaterial.BlendModeEnum.Mix;
		_outline.Modulate = StsColors.halfTransparentWhite;
		_outline.SelfModulate = Colors.Black;
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_visuals, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderS), _hsv.GetShaderParameter(_s), _baseS, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), _baseV, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	protected override void OnPress()
	{
		base.OnPress();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_visuals, "scale", Vector2.One * 0.975f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderS), _hsv.GetShaderParameter(_s), _baseS - 0.1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), _baseV - 0.1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	protected override void OnRelease()
	{
		_isFocused = false;
		_outlineMaterial.BlendMode = CanvasItemMaterial.BlendModeEnum.Mix;
		_outline.Modulate = StsColors.halfTransparentWhite;
		_outline.SelfModulate = Colors.Black;
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_visuals, "scale", Vector2.One, 0.05);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderS), _hsv.GetShaderParameter(_s), _baseS, 0.05);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), _baseV, 0.05);
	}

	private void UpdateShaderS(float value)
	{
		_hsv.SetShaderParameter(_s, value);
	}

	private void UpdateShaderV(float value)
	{
		_hsv.SetShaderParameter(_v, value);
	}
}
