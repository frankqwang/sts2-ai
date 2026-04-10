using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NCubexConstructVfx : Node
{
	private NLaserVfx _laser;

	private Node2D _rings;

	private Node2D _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_laser = _parent.GetNode<NLaserVfx>("LaserBone/Laser");
		_rings = _parent.GetNode<Node2D>("LaserBone/Rings");
		MegaSkin megaSkin = _animController.NewSkin("newSkin");
		MegaSkeleton skeleton = _animController.GetSkeleton();
		MegaSkin skin = skeleton.GetData().FindSkin("moss3");
		megaSkin.AddSkin(skin);
		skeleton.SetSkin(megaSkin);
		_animController.GetAnimationState().SetAnimation("idle_loop");
		_rings.Visible = false;
		_laser.Scale = Vector2.One / _parent.Scale;
		_laser.Visible = false;
		_laser.ResetLaser();
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		string eventName = new MegaEvent(spineEvent).GetData().GetEventName();
		if (!(eventName == "laser_start"))
		{
			if (eventName == "laser_end")
			{
				EndLaser();
			}
		}
		else
		{
			StartLaser();
		}
	}

	private void StartLaser()
	{
		_laser.Visible = true;
		_laser.ExtendLaser(new Vector2(_laser.Position.X - 3000f, _parent.Position.Y));
		Vector2 scale = _rings.Scale;
		_rings.Scale = Vector2.Zero;
		_rings.Visible = true;
		Tween tween = CreateTween();
		tween.TweenProperty(_rings, "scale", scale, 0.4000000059604645).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	private void EndLaser()
	{
		_laser.RetractLaser();
		_rings.Visible = false;
	}
}
