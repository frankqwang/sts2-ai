using Godot;
using MegaCrit.Sts2.Core.Logging;

namespace MegaCrit.Sts2.Core.Nodes.Vfx.Examples;

public partial class NGlowExampleVfx : Node
{
	private WorldEnvironment _env;

	private Tween? _tween;

	public override void _Ready()
	{
		_env = GetNode<WorldEnvironment>("%WorldEnvironment");
		_tween = CreateTween();
		_tween.SetLoops();
		_tween.TweenProperty(_env, "environment:tonemap_exposure", 5f, 1.0).SetEase(Tween.EaseType.InOut).SetTrans(Tween.TransitionType.Cubic);
		_tween.TweenProperty(_env, "environment:tonemap_exposure", 1f, 1.0).SetEase(Tween.EaseType.InOut).SetTrans(Tween.TransitionType.Cubic);
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (inputEvent is InputEventKey inputEventKey && inputEventKey.IsReleased() && inputEventKey.Keycode == Key.Key2)
		{
			if (_tween.IsRunning())
			{
				Log.Info("Pausing Glow");
				_tween.Pause();
			}
			else
			{
				Log.Info("Resuming Rain");
				_tween.Play();
			}
		}
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
	}
}
