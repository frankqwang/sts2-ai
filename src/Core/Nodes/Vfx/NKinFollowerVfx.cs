using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NKinFollowerVfx : Node
{
	private NBasicTrail _trail1;

	private NBasicTrail _trail2;

	private GpuParticles2D _hay;

	private Node2D _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_trail1 = _parent.GetNode<NBasicTrail>("Boomerang1Slot/Trail");
		_trail2 = _parent.GetNode<NBasicTrail>("Boomerang2Slot/Trail");
		_hay = _parent.GetNode<GpuParticles2D>("HaySlot/HayParticles");
		_trail1.Visible = false;
		_trail2.Visible = false;
		_hay.Emitting = false;
		_hay.OneShot = true;
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		switch (new MegaEvent(spineEvent).GetData().GetEventName())
		{
		case "start_trail1":
			StartTrail1();
			break;
		case "end_trail1":
			EndTrail1();
			break;
		case "start_trail2":
			StartTrail2();
			break;
		case "end_trail2":
			EndTrail2();
			break;
		case "start_hay":
			StartHay();
			break;
		}
	}

	private void StartTrail1()
	{
		_trail1.ClearPoints();
		_trail1.Visible = true;
	}

	private void StartTrail2()
	{
		_trail2.ClearPoints();
		_trail2.Visible = true;
	}

	private void EndTrail1()
	{
		_trail1.Visible = false;
	}

	private void EndTrail2()
	{
		_trail2.Visible = false;
	}

	private void StartHay()
	{
		_hay.Restart();
	}
}
