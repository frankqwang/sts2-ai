using Godot;
using Godot.Collections;
using MegaCrit.Sts2.Core.Random;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NDoomSubEmitterVfx : Node2D
{
	[Export(PropertyHint.None, "")]
	private Array<Node2D> _scalableLayers;

	[Export(PropertyHint.None, "")]
	private Array<TextureRect> _spears;

	[Export(PropertyHint.None, "")]
	private Node2D _verticalShrinkingLayer;

	[Export(PropertyHint.None, "")]
	private GpuParticles2D _particlesToKeepDense;

	private Array<Vector2> _baseScales;

	private Array<int> _indeces;

	private float _baseSpearRegionWidth = 140f;

	private float _dumbHackBecauseOfHowTexturerectsWork = -20f;

	private float _rotationHackForSameDumbReason = 4f;

	private int _baseParticleDensity = 300;

	private float _spearFixedHScale = 0.4f;

	private float _spearAngleIntensity = 0.15f;

	private float _minSpearSize = 1f;

	private float _maxSpearSize = 1.5f;

	private float _minSpearTime = 0.3f;

	private float _maxSpearTime = 0.6f;

	private float _outerMargin = 0.2f;

	private float _innerMargin = -0.2f;

	private double _time = 2.0;

	private bool _isOn;

	private float _curScaleX = 1f;

	private Tween? _tween;

	public float CurScaleX
	{
		get
		{
			return _curScaleX;
		}
		set
		{
			_curScaleX = value;
			UpdateWidth(_curScaleX);
		}
	}

	public override void _Ready()
	{
		_baseScales = new Array<Vector2>();
		_indeces = new Array<int>();
		foreach (Node2D scalableLayer in _scalableLayers)
		{
			_baseScales.Add(scalableLayer.Scale);
			_indeces.Add(scalableLayer.GetIndex());
		}
		_isOn = false;
		SetVisibility(isOn: false);
		UpdateWidth(0f);
		ShowOrHide(0f, 0f);
	}

	private void FireSpear(TextureRect textureRect = null)
	{
		Vector2 position = textureRect.Position;
		position.X = Rng.Chaotic.NextFloat(_baseSpearRegionWidth * -0.5f, _baseSpearRegionWidth * 0.5f) + _dumbHackBecauseOfHowTexturerectsWork;
		textureRect.Position = position;
		textureRect.RotationDegrees = textureRect.Position.X * _spearAngleIntensity + _rotationHackForSameDumbReason;
		textureRect.Modulate = Colors.Transparent;
		float x = _spearFixedHScale / _curScaleX;
		textureRect.Scale = new Vector2(x, Rng.Chaotic.NextFloat(_minSpearSize, _maxSpearSize));
		Vector2 vector = new Vector2(x, 0f);
		float num = Rng.Chaotic.NextFloat(_minSpearTime, _maxSpearTime);
		Tween tween = textureRect.CreateTween();
		tween.TweenProperty(textureRect, "scale", vector, num).From(new Vector2(x, Rng.Chaotic.NextFloat(_minSpearSize, _maxSpearSize)));
		if (_isOn)
		{
			tween.TweenCallback(Callable.From(delegate
			{
				FireSpear(textureRect);
			}));
		}
		Tween tween2 = textureRect.CreateTween();
		float a = Rng.Chaotic.NextFloat(0.4f, 0.7f);
		tween2.TweenProperty(textureRect, "modulate", new Color(1f, 1f, 1f, a), 0.20000000298023224).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Quad);
	}

	private void FireAllSpears()
	{
		foreach (TextureRect spear in _spears)
		{
			FireSpear(spear);
		}
	}

	public void ShowOrHide(float widthScale, float tweenTime)
	{
		_isOn = widthScale > 0.1f;
		float num = (_isOn ? 0f : 0.3f);
		float num2 = widthScale;
		Tween.EaseType ease;
		if (_isOn)
		{
			SetVisibility(isOn: true);
			ease = Tween.EaseType.Out;
			int amount = (((int)widthScale * _baseParticleDensity <= 1) ? 1 : ((int)widthScale * _baseParticleDensity));
			_particlesToKeepDense.Amount = amount;
		}
		else
		{
			ease = Tween.EaseType.In;
			num2 = 0f;
			tweenTime = _curScaleX * 0.15f;
		}
		_tween?.Kill();
		_tween = CreateTween();
		_tween.TweenProperty(this, "CurScaleX", num2, tweenTime).SetDelay(num).SetEase(ease)
			.SetTrans(Tween.TransitionType.Cubic);
		if (!_isOn)
		{
			_tween.TweenCallback(Callable.From(delegate
			{
				SetVisibility(isOn: false);
			}));
		}
	}

	private void UpdateWidth(float width)
	{
		_curScaleX = width;
		int num = 0;
		foreach (Node2D scalableLayer in _scalableLayers)
		{
			Vector2 vector = _baseScales[num];
			Vector2 scale = ((scalableLayer != _verticalShrinkingLayer || !(width < 1f)) ? new Vector2(vector.X * _curScaleX, vector.Y) : new Vector2(vector.X * _curScaleX, vector.Y * width / 1f));
			scalableLayer.Scale = scale;
			num++;
		}
	}

	private void SetVisibility(bool isOn)
	{
		int num = 0;
		foreach (Node2D scalableLayer in _scalableLayers)
		{
			if (scalableLayer is GpuParticles2D gpuParticles2D)
			{
				if (isOn)
				{
					gpuParticles2D.Restart();
				}
				else
				{
					gpuParticles2D.Emitting = false;
				}
			}
			else if (isOn)
			{
				MoveChild(scalableLayer, _indeces[num]);
			}
			num++;
		}
		if (isOn)
		{
			FireAllSpears();
		}
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
	}
}
