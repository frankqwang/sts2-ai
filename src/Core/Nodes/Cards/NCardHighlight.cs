using Godot;

namespace MegaCrit.Sts2.Core.Nodes.Cards;

public partial class NCardHighlight : TextureRect
{
	private static readonly StringName _shaderParameterWidth = new StringName("width");

	public static readonly Color playableColor = new Color(0f, 0.957f, 0.988f, 0.98f);

	public static readonly Color gold = new Color(1f, 0.784f, 0f, 0.98f);

	public static readonly Color red = new Color(0.83f, 0f, 0.33f, 0.98f);

	private Tween? _curTween;

	private ShaderMaterial _shaderMaterial;

	public override void _Ready()
	{
		_shaderMaterial = (ShaderMaterial)base.Material;
	}

	public void AnimShow()
	{
		_curTween?.Kill();
		_curTween = CreateTween();
		_curTween.TweenMethod(Callable.From<float>(SetShaderParameter), GetShaderParameter(), 0.075f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
	}

	public void AnimHide()
	{
		_curTween?.Kill();
		_curTween = CreateTween();
		_curTween.TweenMethod(Callable.From<float>(SetShaderParameter), GetShaderParameter(), 0.0, 0.5);
	}

	public void AnimHideInstantly()
	{
		_curTween?.Kill();
		SetShaderParameter(0f);
	}

	public void AnimFlash()
	{
		_curTween?.Kill();
		_curTween = CreateTween();
		_curTween.TweenMethod(Callable.From<float>(SetShaderParameter), GetShaderParameter(), 0.15f, 0.1);
		_curTween.TweenMethod(Callable.From<float>(SetShaderParameter), 0.15f, 0.075f, 0.35).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
	}

	private float GetShaderParameter()
	{
		return _shaderMaterial.GetShaderParameter(_shaderParameterWidth).AsSingle();
	}

	private void SetShaderParameter(float val)
	{
		_shaderMaterial.SetShaderParameter(_shaderParameterWidth, val);
	}
}
