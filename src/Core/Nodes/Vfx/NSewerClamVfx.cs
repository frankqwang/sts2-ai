using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NSewerClamVfx : Node
{
	private const float _coralScaleAmount = 0.2f;

	private const float _maxCoralScale = 1.5f;

	private const float _coralTweenDelay = 0.5f;

	private MegaSprite _megaSprite;

	private GpuParticles2D _deathParticles;

	private GpuParticles2D _buffParticles;

	private GpuParticles2D _chompParticles;

	private Node2D _scaleNode;

	private bool _keyDown;

	private bool _onState;

	public override void _Ready()
	{
		_megaSprite = new MegaSprite(GetParent<Node2D>());
		_megaSprite.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_deathParticles = GetParent().GetNode<GpuParticles2D>("MouthSlot/DeathParticles");
		_buffParticles = GetParent().GetNode<GpuParticles2D>("MouthSlot/BuffParticles");
		_chompParticles = GetParent().GetNode<GpuParticles2D>("MouthSlot/ChompParticles");
		_scaleNode = GetParent().GetNode<Node2D>("CoralScaleBone");
		_deathParticles.Emitting = false;
		_deathParticles.OneShot = true;
		_buffParticles.Emitting = false;
		_chompParticles.OneShot = true;
		_chompParticles.Emitting = false;
		_scaleNode.Scale = new Vector2(0.1f, 0.1f);
	}

	private void ScaleCoralTo(float targetScale)
	{
		Vector2 vector = new Vector2(targetScale - 0.2f, targetScale - 0.2f);
		Vector2 vector2 = new Vector2(targetScale, targetScale);
		Tween tween = CreateTween();
		tween.SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Elastic);
		tween.TweenProperty(_scaleNode, "scale", vector2, 0.5).From(vector).SetDelay(0.5);
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		switch (new MegaEvent(spineEvent).GetData().GetEventName())
		{
		case "death_explode":
			OnDeathStart();
			break;
		case "darkness_start":
			OnDarknessStart();
			break;
		case "darkness_end":
			OnDarknessEnd();
			break;
		case "chomp":
			OnChomp();
			break;
		case "grow":
			OnGrow();
			break;
		}
	}

	private void OnDeathStart()
	{
		_deathParticles.Restart();
	}

	private void OnDeathEnd()
	{
		_deathParticles.Emitting = false;
	}

	private void OnDarknessStart()
	{
		_buffParticles.Restart();
	}

	private void OnDarknessEnd()
	{
		_buffParticles.Emitting = false;
	}

	private void OnChomp()
	{
		_chompParticles.Restart();
	}

	private void OnGrow()
	{
		float num = _scaleNode.Scale.X + 0.2f;
		if (num >= 1.5f)
		{
			num = 1.5f;
		}
		ScaleCoralTo(num);
	}
}
