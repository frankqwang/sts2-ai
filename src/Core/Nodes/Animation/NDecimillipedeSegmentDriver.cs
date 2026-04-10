using Godot;

namespace MegaCrit.Sts2.Core.Nodes.Animation;

public partial class NDecimillipedeSegmentDriver : Node2D
{
	[Export(PropertyHint.None, "")]
	private bool _leftSegment;

	private float _speed;

	private float _magnitude;

	private Vector2 _originPos;

	private FastNoiseLite _noise = new FastNoiseLite();

	private float _time;

	private Vector2 _decimillipedeStrikeOffset = Vector2.Zero;

	private Tween? _attackTween;

	public override void _Ready()
	{
		_originPos = base.Position;
		_speed = (_leftSegment ? 0.1f : 0.05f);
		_magnitude = (_leftSegment ? 250f : 300f);
		_noise.NoiseType = FastNoiseLite.NoiseTypeEnum.Perlin;
		_noise.Frequency = 1f;
	}

	public override void _Process(double delta)
	{
		_time += (float)delta;
		float num = _time * _speed + (_leftSegment ? 0.25f : 0f);
		Vector2 vector = new Vector2(_noise.GetNoise1D(num), _noise.GetNoise1D(num + 0.25f)) * _magnitude;
		base.Position = _originPos + vector + _decimillipedeStrikeOffset;
	}

	public void AttackShake()
	{
		_attackTween?.Kill();
		_attackTween = CreateTween();
		_attackTween.TweenProperty(this, "_decimillipedeStrikeOffset", Vector2.Left * 100f, 0.4).SetEase(Tween.EaseType.InOut).SetTrans(Tween.TransitionType.Sine);
		_attackTween.TweenProperty(this, "_decimillipedeStrikeOffset", Vector2.Right * 100f, 0.10000000149011612).SetEase(Tween.EaseType.InOut).SetTrans(Tween.TransitionType.Sine);
		_attackTween.TweenProperty(this, "_decimillipedeStrikeOffset", Vector2.Zero, 0.75).SetEase(Tween.EaseType.InOut).SetTrans(Tween.TransitionType.Sine);
	}
}
