using Godot;
using Godot.Collections;
using MegaCrit.Sts2.Core.Random;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NRestSiteFireVfx : Node2D
{
	[Export(PropertyHint.None, "")]
	private float _minFlickerScale = 0.85f;

	[Export(PropertyHint.None, "")]
	private float _maxFlickerScale = 1.05f;

	[Export(PropertyHint.None, "")]
	private float _minFlickerTime = 0.3f;

	[Export(PropertyHint.None, "")]
	private float _maxFlickerTime = 0.5f;

	[Export(PropertyHint.None, "")]
	private float _minSkew = -0.1f;

	[Export(PropertyHint.None, "")]
	private float _maxSkew = 0.1f;

	[Export(PropertyHint.None, "")]
	private float _minSkewTime = 0.8f;

	[Export(PropertyHint.None, "")]
	private float _maxSkewTime = 1.5f;

	[Export(PropertyHint.None, "")]
	private float _extinguishTime = 0.2f;

	[Export(PropertyHint.None, "")]
	private bool _enabled = true;

	[Export(PropertyHint.None, "")]
	private Array<CpuParticles2D> _cpuGlowParticles = new Array<CpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _gpuSparkParticles = new Array<GpuParticles2D>();

	private Vector2 _baseScale;

	private float _baseSkew;

	private Tween? _scaleTweenRef;

	public override void _Ready()
	{
		if (_enabled)
		{
			_baseScale = base.Scale;
			_baseSkew = base.Skew;
			Flicker();
			Sway();
		}
	}

	private void Flicker()
	{
		Vector2 vector = new Vector2(_baseScale.X, Rng.Chaotic.NextFloat(_baseScale.Y * _minFlickerScale, _baseScale.Y));
		Vector2 vector2 = new Vector2(_baseScale.X, Rng.Chaotic.NextFloat(_baseScale.Y, _baseScale.Y * _maxFlickerScale));
		Tween tween = CreateTween();
		tween.TweenProperty(this, "scale", vector, Rng.Chaotic.NextFloat(_minFlickerTime, _maxFlickerTime)).SetTrans(Tween.TransitionType.Quad).SetEase(Tween.EaseType.InOut);
		tween.TweenProperty(this, "scale", vector2, Rng.Chaotic.NextFloat(_minFlickerTime, _maxFlickerTime)).SetTrans(Tween.TransitionType.Quad).SetEase(Tween.EaseType.InOut);
		tween.TweenCallback(Callable.From(Flicker));
		_scaleTweenRef = tween;
	}

	private void Sway()
	{
		float num = Rng.Chaotic.NextFloat(_baseSkew + _minSkew, _baseSkew);
		float num2 = Rng.Chaotic.NextFloat(_baseSkew, _baseSkew + _maxSkew);
		Tween tween = CreateTween();
		tween.TweenProperty(this, "skew", num, Rng.Chaotic.NextFloat(_minSkewTime, _maxSkewTime)).SetTrans(Tween.TransitionType.Sine).SetEase(Tween.EaseType.InOut);
		tween.TweenProperty(this, "skew", num2, Rng.Chaotic.NextFloat(_minSkewTime, _maxSkewTime)).SetTrans(Tween.TransitionType.Sine).SetEase(Tween.EaseType.InOut);
		tween.TweenCallback(Callable.From(Sway));
	}

	public void Extinguish()
	{
		Tween tween = CreateTween().SetParallel();
		foreach (CpuParticles2D cpuGlowParticle in _cpuGlowParticles)
		{
			cpuGlowParticle.Emitting = false;
			tween.TweenProperty(cpuGlowParticle, "scale", Vector2.Zero, _extinguishTime).SetTrans(Tween.TransitionType.Back).SetEase(Tween.EaseType.In);
		}
		foreach (GpuParticles2D gpuSparkParticle in _gpuSparkParticles)
		{
			gpuSparkParticle.Emitting = false;
		}
		_scaleTweenRef?.Kill();
		Tween tween2 = CreateTween();
		tween2.TweenProperty(this, "scale", Vector2.Zero, _extinguishTime).SetTrans(Tween.TransitionType.Back).SetEase(Tween.EaseType.In);
	}

	public override void _ExitTree()
	{
		_scaleTweenRef?.Kill();
	}
}
