using Godot;
using MegaCrit.Sts2.Core.Random;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NKinPriestBeamVfx : Node2D
{
	private const float _beamMaxLengthScale = 4f;

	private const float _startRotation = 1f;

	private const float _endRotation = -1f;

	private Sprite2D _beam;

	private Node2D _beamHolder;

	private GpuParticles2D _staticParticles;

	private Vector2 _baseBeamScale;

	private Tween? _lengthTween;

	private Tween? _rotationTween;

	public override void _Ready()
	{
		_beam = GetNode<Sprite2D>("BeamHolder/Beam");
		_staticParticles = GetNode<GpuParticles2D>("BeamHolder/StaticParticles");
		_beamHolder = GetNode<Node2D>("BeamHolder");
		_baseBeamScale = _beam.Scale;
		_staticParticles.Emitting = false;
		_staticParticles.Visible = false;
		_beamHolder.Visible = false;
	}

	public override void _Process(double delta)
	{
		Vector2 vector = new Vector2(Rng.Chaotic.NextFloat(-0.05f, 0.05f), Rng.Chaotic.NextFloat(-0.7f, 0.7f));
		_beam.Scale = _baseBeamScale + vector;
		Color modulate = base.Modulate;
		modulate.A = Rng.Chaotic.NextFloat(0.8f, 1f);
		base.Modulate = modulate;
	}

	public void Fire()
	{
		_staticParticles.Restart();
		_staticParticles.Visible = true;
		_beamHolder.Visible = true;
		_rotationTween = CreateTween();
		base.RotationDegrees = 1f;
		_rotationTween.TweenProperty(this, "rotation_degrees", -1f, 0.800000011920929).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		_beamHolder.Scale = Vector2.One;
		_lengthTween = CreateTween();
		_lengthTween.TweenProperty(_beamHolder, "scale:x", 4f, 0.3799999952316284).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_lengthTween.Chain().TweenProperty(_beamHolder, "scale:x", 0.5, 0.6000000238418579).SetEase(Tween.EaseType.In)
			.SetTrans(Tween.TransitionType.Expo);
		_lengthTween.TweenCallback(Callable.From(OnTweenComplete));
	}

	private void OnTweenComplete()
	{
		_rotationTween.Kill();
		_lengthTween.Kill();
		_staticParticles.Emitting = false;
		_staticParticles.Visible = false;
		_beamHolder.Visible = false;
	}

	public override void _ExitTree()
	{
		_lengthTween?.Kill();
		_rotationTween?.Kill();
	}
}
