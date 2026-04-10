using System.Linq;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;

public partial class NChoiceSelectionSkipButton : NButton
{
	private static readonly StringName _v = new StringName("v");

	private TextureRect _image;

	private MegaLabel _label;

	private string? _optionName;

	private Tween? _animInTween;

	private Vector2 _showPosition;

	private static readonly Vector2 _animOffsetPosition = new Vector2(0f, -50f);

	private Tween? _currentTween;

	private ShaderMaterial _hsv;

	private Variant _hsvDefault = 0.9;

	private Variant _hsvHover = 1.1;

	private Variant _hsvDown = 0.7;

	private static readonly Vector2 _defaultScale = Vector2.One;

	private static readonly Vector2 _hoverScale = Vector2.One * 1.05f;

	private static readonly Vector2 _downScale = Vector2.One * 0.95f;

	protected override string[] Hotkeys => new string[1] { MegaInput.cancel };

	public override void _Ready()
	{
		ConnectSignals();
		_image = GetNode<TextureRect>("Image");
		_label = GetNode<MegaLabel>("Label");
		_hsv = (ShaderMaterial)_image.Material;
		_showPosition = base.Position;
		if (_optionName != null)
		{
			_label.SetTextAutoSize(_optionName);
		}
		_controllerHotkeyIcon.Texture = NInputManager.Instance.GetHotkeyIcon(Hotkeys.First());
	}

	public void AnimateIn()
	{
		_animInTween?.Kill();
		_animInTween = CreateTween().SetParallel();
		_animInTween.TweenProperty(this, "modulate:a", 1f, 0.4).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
			.From(0f);
		_animInTween.TweenProperty(this, "position", _showPosition, 0.4).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back)
			.From(_showPosition + _animOffsetPosition);
	}

	protected override void OnPress()
	{
		_currentTween?.Kill();
		_currentTween = CreateTween().SetParallel();
		_currentTween.TweenMethod(Callable.From<float>(UpdateShaderParam), _hsvHover, _hsvDown, 0.20000000298023224).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_currentTween.TweenProperty(this, "scale", _downScale, 0.20000000298023224).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	protected override void OnFocus()
	{
		_currentTween?.Kill();
		base.Scale = _hoverScale;
		_hsv.SetShaderParameter(_v, _hsvHover);
	}

	protected override void OnUnfocus()
	{
		_currentTween?.Kill();
		_currentTween = CreateTween().SetParallel();
		_currentTween.TweenMethod(Callable.From<float>(UpdateShaderParam), _hsv.GetShaderParameter(_v), _hsvDefault, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_currentTween.TweenProperty(this, "scale", _defaultScale, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	private void UpdateShaderParam(float value)
	{
		_hsv.SetShaderParameter(_v, value);
	}
}
