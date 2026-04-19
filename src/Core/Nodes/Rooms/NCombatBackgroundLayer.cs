using Godot;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Rooms;

public partial class NCombatBackgroundLayer : Control
{
	private Control _visual;

	private Control _phobiaModeVisual;

	public override void _Ready()
	{
		_visual = GetNode<Control>("Visual");
		_phobiaModeVisual = GetNode<Control>("PhobiaModeVisual");
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
		_phobiaModeVisual.Visible = SaveManager.Instance.PrefsSave.PhobiaMode;
		_visual.Visible = !_phobiaModeVisual.Visible;
	}
}
