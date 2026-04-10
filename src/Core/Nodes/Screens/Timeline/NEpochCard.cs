using Godot;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Timeline;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Timeline;

public partial class NEpochCard : Control
{
	private TextureRect _glow;

	private TextureRect _mask;

	private TextureRect _portrait;

	private bool _isHovered;

	private bool _isHoverable = true;

	private bool _isHeld;

	private bool _isWigglyUnlockPreviewMode;

	private Tween? _glowTween;

	private Vector2 _targetScale;

	private float _time;

	private float _noiseSpeed = 0.25f;

	private FastNoiseLite _noise;

	private Tween? _denyTween;

	private Tween? _transparencyTween;

	private Tween? _scaleTween;

	private Color _blueGlowColor = new Color("2de5ff80");

	private Color _goldGlowColor = new Color("ffd92e80");

	public void Init(EpochModel epochModel)
	{
		_glow = GetNode<TextureRect>("%GlowPlaceholder");
		_portrait = GetNode<TextureRect>("%Portrait");
		_portrait.Texture = epochModel.BigPortrait;
		base.Scale = Vector2.One;
	}

	public override void _Ready()
	{
		_mask = GetNode<TextureRect>("%Mask");
	}

	public override void _Process(double delta)
	{
		if (_isWigglyUnlockPreviewMode)
		{
			_time += _noiseSpeed * (float)delta;
			float x = 42f * _noise.GetNoise2D(_time, 0f);
			float y = 42f * _noise.GetNoise2D(0f, _time);
			base.Position = new Vector2(x, y) - GetPivotOffset();
		}
	}

	public void SetToWigglyUnlockPreviewMode()
	{
		_noise = new FastNoiseLite();
		_noise.Frequency = 0.2f;
		_noise.SetSeed(Rng.Chaotic.NextInt());
		_isWigglyUnlockPreviewMode = true;
		base.MouseFilter = MouseFilterEnum.Ignore;
		base.Scale = Vector2.One * 1.2f;
	}

	private void GlowFlash()
	{
		base.Modulate = new Color(1f, 1f, 1f, 0.75f);
		_glow.Modulate = Colors.Gold;
		_glowTween?.Kill();
		_glowTween = CreateTween().SetParallel();
		_glowTween.TweenProperty(_glow, "modulate:a", 0.5f, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_glowTween.TweenProperty(_glow, "scale", Vector2.One, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
			.From(Vector2.One * 1.5f);
	}
}
