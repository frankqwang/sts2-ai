using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NEntomancerVfx : Node
{
	private GpuParticles2D _swarmParticles;

	private GpuParticles2D _attackingBugParticles;

	private Node2D _swarmTargetNode;

	private Vector2 _basePosition;

	private Tween? _swarmTween;

	private bool _swarming;

	private Node2D _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_swarmParticles = _parent.GetNode<GpuParticles2D>("SwarmParticles");
		_attackingBugParticles = _parent.GetNode<GpuParticles2D>("SwarmParticles/AttackingBugParticles");
		_swarmTargetNode = _parent.GetNode<Node2D>("SwarmTargetNode");
		_basePosition = _swarmParticles.Position;
		_attackingBugParticles.Emitting = false;
	}

	private void LaunchSwarm()
	{
		_swarming = true;
		_swarmTween = CreateTween();
		_attackingBugParticles.Emitting = true;
		_swarmTween.SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		_swarmTween.TweenProperty(_swarmParticles, "position", _swarmTargetNode.Position, 1.0);
		_swarmTween.SetEase(Tween.EaseType.InOut).SetTrans(Tween.TransitionType.Cubic);
		_swarmTween.TweenProperty(_swarmParticles, "position", _basePosition, 1.5).SetDelay(0.0);
		_swarmTween.TweenCallback(Callable.From(CompleteSwarmAttack)).SetDelay(0.009999999776482582);
	}

	private void CompleteSwarmAttack()
	{
		_attackingBugParticles.Emitting = false;
		_swarming = false;
	}

	private void CancelSwarmAttack()
	{
		if (_swarming)
		{
			_swarmTween.Kill();
			_swarmParticles.Position = _basePosition;
			CompleteSwarmAttack();
		}
	}

	private void TurnOffSwarm()
	{
		_swarmParticles.Emitting = false;
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		string eventName = new MegaEvent(spineEvent).GetData().GetEventName();
		if (!(eventName == "launch_swarm"))
		{
			if (eventName == "turn_off_swarm")
			{
				TurnOffSwarm();
			}
		}
		else
		{
			LaunchSwarm();
		}
	}

	public override void _ExitTree()
	{
		_swarmTween?.Kill();
	}
}
