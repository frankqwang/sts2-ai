using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NFuzzyWurmCrawlerSpitTrailVfx : Node
{
	private bool _isKeyDown;

	private CpuParticles2D _trailParticles;

	private Node2D _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_animController.ConnectAnimationStarted(Callable.From<GodotObject, GodotObject, GodotObject>(OnAnimationStart));
		_trailParticles = _parent.GetNode<CpuParticles2D>("SpitParticlesBone/TrailParticles");
		_trailParticles.Emitting = false;
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		string eventName = new MegaEvent(spineEvent).GetData().GetEventName();
		if (!(eventName == "launch_start"))
		{
			if (eventName == "launch_end")
			{
				TurnOffTrail();
			}
		}
		else
		{
			TurnOnTrail();
		}
	}

	private void OnAnimationStart(GodotObject spineSprite, GodotObject animationState, GodotObject trackEntry)
	{
		if (new MegaAnimationState(animationState).GetCurrent(0).GetAnimation().GetName() != "attack")
		{
			TurnOffTrail();
		}
	}

	private void TurnOnTrail()
	{
		_trailParticles.Restart();
	}

	private void TurnOffTrail()
	{
		_trailParticles.Emitting = false;
	}
}
