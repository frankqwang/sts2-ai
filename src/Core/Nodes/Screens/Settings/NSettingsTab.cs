using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NSettingsTab : NButton
{
	private static readonly StringName _v = new StringName("v");

	private bool _isSelected;

	private TextureRect _outline;

	private TextureRect _image;

	private MegaLabel _label;

	private ShaderMaterial _hsv;

	private Tween? _tween;

	private const float _defaultV = 0.9f;

	private const float _hoverV = 1.2f;

	private const float _unhoverDuration = 0.5f;

	public override void _Ready()
	{
		ConnectSignals();
		_outline = GetNode<TextureRect>("Outline");
		_image = GetNode<TextureRect>("TabImage");
		_label = GetNode<MegaLabel>("Label");
		_hsv = (ShaderMaterial)_image.Material;
		_label.Modulate = StsColors.halfTransparentCream;
	}

	public void ForceTabPressed()
	{
		EmitSignalReleased(this);
	}

	public void Select()
	{
		if (!_isSelected)
		{
			_isSelected = true;
			_tween?.Kill();
			_label.Modulate = StsColors.cream;
			_image.Modulate = Colors.White;
			_outline.Visible = true;
		}
	}

	public void Deselect()
	{
		if (_isSelected)
		{
			_isSelected = false;
			_tween?.Kill();
			_outline.Visible = false;
			_label.Modulate = StsColors.halfTransparentCream;
		}
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_tween?.Kill();
		_label.Modulate = StsColors.gold;
		_hsv.SetShaderParameter(_v, 1.2f);
		base.Scale = Vector2.One * 1.05f;
	}

	protected override void OnUnfocus()
	{
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		Color color = (_isSelected ? StsColors.cream : StsColors.halfTransparentCream);
		_tween.TweenProperty(_label, "modulate", color, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderParam), 1.2f, 0.9f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	protected override void OnEnable()
	{
		_tween?.Kill();
		base.Modulate = Colors.White;
	}

	protected override void OnDisable()
	{
		_tween?.Kill();
		base.Modulate = Colors.DimGray;
	}

	private void UpdateShaderParam(float newV)
	{
		_hsv.SetShaderParameter(_v, newV);
	}

	public void SetLabel(string text)
	{
		_label.SetTextAutoSize(text);
	}
}
