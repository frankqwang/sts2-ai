using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NFlyconidSporesVfx : Node
{
	private CpuParticles2D _frailSpores;

	private CpuParticles2D _vulnerableSpores;

	private Node2D _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_animController.ConnectAnimationStarted(Callable.From<GodotObject, GodotObject, GodotObject>(OnAnimationStart));
		_frailSpores = _parent.GetNode<CpuParticles2D>("SporesAttach/FlyconidSporesVfx/FrailSpores");
		_frailSpores.Emitting = false;
		_vulnerableSpores = _parent.GetNode<CpuParticles2D>("SporesAttach/FlyconidSporesVfx/VulnerableSpores");
		_vulnerableSpores.Emitting = false;
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		string eventName = new MegaEvent(spineEvent).GetData().GetEventName();
		if (!(eventName == "cast_start"))
		{
			if (eventName == "cast_end")
			{
				_frailSpores.Emitting = false;
				_vulnerableSpores.Emitting = false;
			}
		}
		else
		{
			_frailSpores.Restart();
			_vulnerableSpores.Restart();
		}
	}

	public void SetSporeTypeIsVulnerable(bool isVulnerable)
	{
		_frailSpores.Visible = !isVulnerable;
		_vulnerableSpores.Visible = isVulnerable;
	}

	private void OnAnimationStart(GodotObject spineSprite, GodotObject animationState, GodotObject trackEntry)
	{
		if (new MegaAnimationState(animationState).GetCurrent(0).GetAnimation().GetName() != "attack")
		{
			_frailSpores.Emitting = false;
			_frailSpores.Emitting = false;
		}
	}
}
