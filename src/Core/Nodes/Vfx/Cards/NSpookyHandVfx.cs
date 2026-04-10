using Godot;
using MegaCrit.Sts2.Core.Random;

namespace MegaCrit.Sts2.Core.Nodes.Vfx.Cards;

public partial class NSpookyHandVfx : Control
{
	private const float _pauseDuration = 0.05f;

	private float _elapsedPauseTime;

	private int _pauseCounter;

	private bool _isPaused;

	private float _timer;

	private int _totalPauses;

	private float _canPauseTimer;

	private float _intensity;

	private float _speed;

	private float _duration;

	private float _originalRotation;

	private Vector2 _targetScale;

	public override void _Ready()
	{
		_totalPauses = Rng.Chaotic.NextInt(2, 7);
		_canPauseTimer = Rng.Chaotic.NextFloat(0.5f, 1.2f);
		_speed = Rng.Chaotic.NextFloat(3f, 5f);
		_intensity = Rng.Chaotic.NextFloat(0.1f, 0.3f);
		_originalRotation = base.Rotation;
		_targetScale = base.Scale;
		AnimateIn();
	}

	private void AnimateIn()
	{
		base.Scale = Vector2.Zero;
		Tween tween = CreateTween().SetParallel();
		base.Modulate = new Color(base.Modulate.R + Rng.Chaotic.NextFloat(-0.2f, 0.2f), base.Modulate.G, base.Modulate.B);
		tween.TweenInterval(Rng.Chaotic.NextDouble(0.0, 0.4));
		tween.Chain();
		tween.TweenProperty(this, "scale", _targetScale, Rng.Chaotic.NextDouble(0.4, 0.5)).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Spring);
		tween.Chain();
		tween.TweenInterval(Rng.Chaotic.NextDouble(0.3, 0.6));
		tween.Chain();
		double duration = Rng.Chaotic.NextDouble(0.4, 0.6);
		tween.TweenProperty(this, "scale", _targetScale * 0.5f, duration).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Back);
		tween.TweenProperty(this, "modulate:a", 0f, duration).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Quad);
	}

	public override void _Process(double delta)
	{
		float num = (float)delta;
		_duration += num * _speed;
		_canPauseTimer -= num;
		if (_isPaused)
		{
			_elapsedPauseTime += num;
			if (_elapsedPauseTime >= 0.05f)
			{
				_isPaused = false;
				_elapsedPauseTime = 0f;
				_pauseCounter++;
			}
			base.Rotation = _originalRotation + Mathf.Sin(_duration) * _intensity * 0.5f;
		}
		else
		{
			base.Rotation = _originalRotation + Mathf.Sin(_duration) * _intensity;
		}
		_timer += num;
		if (_canPauseTimer < 0f && _pauseCounter < _totalPauses)
		{
			_isPaused = true;
			_timer = 0f;
		}
	}
}
