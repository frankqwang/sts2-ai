using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NTheInsatiableVfx : Node
{
	[Export(PropertyHint.None, "")]
	private CpuParticles2D[] _continuousParticles;

	private CpuParticles2D _salivaFountainParticles;

	private CpuParticles2D _salivaDroolParticles;

	private CpuParticles2D _salivaCloudParticles;

	private GpuParticles2D _baseBlastParticles;

	private Node2D _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_animController.ConnectAnimationStarted(Callable.From<GodotObject, GodotObject, GodotObject>(OnAnimationStart));
		_salivaFountainParticles = _parent.GetNode<CpuParticles2D>("SalivaSlotNode/SalivaFountainParticles");
		_salivaDroolParticles = _parent.GetNode<CpuParticles2D>("SalivaSlotNode/SalivaDroolParticles");
		_salivaCloudParticles = _parent.GetNode<CpuParticles2D>("SalivaSlotNode/SalivaCloudParticles");
		_baseBlastParticles = _parent.GetNode<GpuParticles2D>("BaseBlastSlot/BaseBlastParticles");
		_salivaFountainParticles.Emitting = false;
		_salivaDroolParticles.Emitting = false;
		_salivaCloudParticles.Emitting = false;
		_baseBlastParticles.Emitting = false;
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		string eventName = new MegaEvent(spineEvent).GetData().GetEventName();
		if (eventName == null)
		{
			return;
		}
		switch (eventName.Length)
		{
		case 9:
			switch (eventName[1])
			{
			case 'r':
				if (eventName == "drool_end")
				{
					TurnOffDrool();
				}
				break;
			case 'e':
				if (eventName == "death_end")
				{
					TurnOffContinuousParticles();
				}
				break;
			}
			break;
		case 12:
			if (eventName == "saliva_start")
			{
				TurnOnSaliva();
			}
			break;
		case 10:
			if (eventName == "saliva_end")
			{
				TurnOffSaliva();
			}
			break;
		case 11:
			if (eventName == "drool_start")
			{
				TurnOnDrool();
			}
			break;
		case 16:
			if (eventName == "base_blast_start")
			{
				TurnOnBaseBlast();
			}
			break;
		case 14:
			if (eventName == "base_blast_end")
			{
				TurnOffBaseBlast();
			}
			break;
		case 13:
		case 15:
			break;
		}
	}

	private void TurnOnSaliva()
	{
		_salivaFountainParticles.Restart();
		_salivaCloudParticles.Restart();
	}

	private void TurnOffSaliva()
	{
		_salivaFountainParticles.Emitting = false;
		_salivaCloudParticles.Emitting = false;
	}

	private void TurnOnDrool()
	{
		_salivaDroolParticles.Restart();
	}

	private void TurnOffDrool()
	{
		_salivaDroolParticles.Emitting = false;
	}

	private void TurnOnBaseBlast()
	{
		_baseBlastParticles.Emitting = true;
	}

	private void TurnOffBaseBlast()
	{
		_baseBlastParticles.Emitting = false;
	}

	private void TurnOffContinuousParticles()
	{
		CpuParticles2D[] continuousParticles = _continuousParticles;
		foreach (CpuParticles2D cpuParticles2D in continuousParticles)
		{
			cpuParticles2D.Emitting = false;
		}
	}

	private void OnAnimationStart(GodotObject spineSprite, GodotObject animationState, GodotObject trackEntry)
	{
		if (new MegaAnimationState(animationState).GetCurrent(0).GetAnimation().GetName() != "attack_thrash")
		{
			TurnOffBaseBlast();
		}
		if (new MegaAnimationState(animationState).GetCurrent(0).GetAnimation().GetName() != "salivate")
		{
			TurnOffSaliva();
		}
	}
}
