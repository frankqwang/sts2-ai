using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;

namespace MegaCrit.Sts2.Core.Nodes.Ftue;

public partial class NFtue : Control, IScreenContext
{
	public Control? DefaultFocusedControl => this;

	public override void _EnterTree()
	{
		base.FocusMode = FocusModeEnum.All;
		base.FocusNeighborBottom = GetPath();
		base.FocusNeighborTop = GetPath();
		base.FocusNeighborLeft = GetPath();
		base.FocusNeighborRight = GetPath();
		NHotkeyManager.Instance.AddBlockingScreen(this);
	}

	public override void _ExitTree()
	{
		NHotkeyManager.Instance.RemoveBlockingScreen(this);
	}

	protected void CloseFtue()
	{
		NModalContainer.Instance.Clear();
		this.QueueFreeSafely();
	}
}
