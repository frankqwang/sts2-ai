using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NCeremonialBeastVfx : Node, IDeathDelayer
{
	[Export(PropertyHint.None, "")]
	private GpuParticles2D _deathParticles;

	[Export(PropertyHint.None, "")]
	private CpuParticles2D _energyParticlesFront;

	[Export(PropertyHint.None, "")]
	private CpuParticles2D _energyParticlesBack;

	[Export(PropertyHint.None, "")]
	private Node2D _plowStartTarget;

	[Export(PropertyHint.None, "")]
	private Node2D _plowEndTarget;

	private Node2D _parent;

	private MegaSprite _animController;

	private Vector2 _globalPlowTarget;

	private Vector2 _globalPlowEndTarget;

	private readonly TaskCompletionSource _deathTask = new TaskCompletionSource();

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_deathParticles.OneShot = true;
		_deathParticles.Emitting = false;
		_energyParticlesBack.Emitting = true;
		_energyParticlesFront.Emitting = true;
		_globalPlowTarget = _plowStartTarget.GlobalPosition;
		_globalPlowEndTarget = _plowEndTarget.GlobalPosition;
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		switch (new MegaEvent(spineEvent).GetData().GetEventName())
		{
		case "turnOffEnergy":
			TurnOffEnergyParticles();
			break;
		case "turnOnEnergy":
			TurnOnEnergyParticles();
			break;
		case "deathParticles":
			TurnOnDeathParticles();
			break;
		case "plowStart":
			OnPlowStart();
			break;
		case "plowEnd":
			OnPlowEnd();
			break;
		}
	}

	public Task GetDelayTask()
	{
		return _deathTask.Task;
	}

	private void TurnOnDeathParticles()
	{
		_deathParticles.Restart();
		TaskHelper.RunSafely(FinishTaskWhenDeathParticlesFinished());
	}

	private async Task FinishTaskWhenDeathParticlesFinished()
	{
		await ToSignal(_deathParticles, CpuParticles2D.SignalName.Finished);
		_deathTask.SetResult();
	}

	private void TurnOnEnergyParticles()
	{
		_energyParticlesFront.Emitting = true;
		_energyParticlesBack.Emitting = true;
	}

	private void TurnOffEnergyParticles()
	{
		_energyParticlesFront.Emitting = false;
		_energyParticlesBack.Emitting = false;
	}

	private void OnPlowStart()
	{
		_plowStartTarget.GlobalPosition = _globalPlowTarget;
	}

	private void OnPlowEnd()
	{
		_plowEndTarget.GlobalPosition = _globalPlowEndTarget;
	}
}
