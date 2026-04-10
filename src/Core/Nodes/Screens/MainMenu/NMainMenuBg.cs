using Godot;

namespace MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;

public partial class NMainMenuBg : Control
{
	private Window _window;

	private Control _bg;

	private Node2D _logo;

	private Tween? _logoTween;

	private static readonly Vector2 _defaultBgScale = Vector2.One * 1.01f;

	private const float _bgScaleRatioThreshold = 1.5f;

	public override void _Ready()
	{
		_bg = GetNode<Control>("BgContainer");
		_logo = GetNode<Node2D>("%Logo");
		_window = GetTree().Root;
		_window.Connect(Viewport.SignalName.SizeChanged, Callable.From(OnWindowChange));
	}

	private void OnWindowChange()
	{
		ScaleBgIfNarrow((float)_window.Size.X / (float)_window.Size.Y);
	}

	private void ScaleBgIfNarrow(float ratio)
	{
		if (ratio < 1.5f)
		{
			_bg.Scale = Vector2.One * 1.04f;
		}
		else
		{
			_bg.Scale = _defaultBgScale;
		}
	}

	public void HideLogo()
	{
		_logoTween?.Kill();
		_logoTween = CreateTween();
		_logoTween.TweenProperty(_logo, "modulate:a", 0f, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
	}

	public void ShowLogo()
	{
		_logoTween?.Kill();
		_logoTween = CreateTween();
		_logoTween.TweenProperty(_logo, "modulate:a", 1f, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
	}
}
