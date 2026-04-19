using Godot;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Animation;

public partial class NPhobiaAnimationToggler : Node
{
	[Export(PropertyHint.None, "")]
	private AnimationPlayer? _animationPlayer;

	public override void _Ready()
	{
		base._Ready();
		UpdatePhobiaMode();
	}

	public override void _EnterTree()
	{
		NGame.Instance?.Connect(NGame.SignalName.PhobiaModeToggled, Callable.From(UpdatePhobiaMode));
	}

	public override void _ExitTree()
	{
		NGame.Instance?.Disconnect(NGame.SignalName.PhobiaModeToggled, Callable.From(UpdatePhobiaMode));
	}

	private void UpdatePhobiaMode()
	{
		if (_animationPlayer != null)
		{
			_animationPlayer.Active = !SaveManager.Instance.PrefsSave.PhobiaMode;
		}
	}
}
