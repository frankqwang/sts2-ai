using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NArchitectVfx : Node
{
	private Node2D _parent;

	private MegaSprite _animController;

	private NBasicTrail _innerTrail;

	private NBasicTrail _outerTrail;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_innerTrail = _parent.GetNode<NBasicTrail>("TrailSlot/TrailInner");
		_outerTrail = _parent.GetNode<NBasicTrail>("TrailSlot/TrailOuter");
		_animController.GetAnimationState().SetAnimation("idle_loop");
		_animController.GetAnimationState().SetAnimation("_tracks/head_normal", loop: true, 1);
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		string eventName = new MegaEvent(spineEvent).GetData().GetEventName();
		if (!(eventName == "trail_start"))
		{
			if (eventName == "trail_end")
			{
				EndTrail();
			}
		}
		else
		{
			StartTrail();
		}
	}

	private void StartTrail()
	{
		_innerTrail.Visible = true;
		_outerTrail.Visible = true;
		_innerTrail.ClearPoints();
		_outerTrail.ClearPoints();
	}

	private void EndTrail()
	{
		_innerTrail.Visible = false;
		_outerTrail.Visible = false;
	}
}
