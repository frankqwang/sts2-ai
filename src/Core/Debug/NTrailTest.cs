using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Vfx;

namespace MegaCrit.Sts2.Core.Debug;

public partial class NTrailTest : Control
{
	public override void _Ready()
	{
		DelaySpawn();
	}

	private async Task DelaySpawn()
	{
		await Task.Delay(100);
		Control node = GetNode<Control>("Ironclad");
		NCardTrailVfx child = NCardTrailVfx.Create(node, SceneHelper.GetScenePath("vfx/card_trail_ironclad"));
		GetParent().AddChildSafely(child);
		Control node2 = GetNode<Control>("Silent");
		child = NCardTrailVfx.Create(node2, SceneHelper.GetScenePath("vfx/card_trail_silent"));
		GetParent().AddChildSafely(child);
		Control node3 = GetNode<Control>("Defect");
		child = NCardTrailVfx.Create(node3, SceneHelper.GetScenePath("vfx/card_trail_defect"));
		GetParent().AddChildSafely(child);
		Control node4 = GetNode<Control>("Regent");
		child = NCardTrailVfx.Create(node4, SceneHelper.GetScenePath("vfx/card_trail_regent"));
		GetParent().AddChildSafely(child);
		Control node5 = GetNode<Control>("Binder");
		child = NCardTrailVfx.Create(node5, SceneHelper.GetScenePath("vfx/card_trail_necrobinder"));
		GetParent().AddChildSafely(child);
	}

	public override void _Process(double delta)
	{
		Vector2 mousePosition = GetViewport().GetMousePosition();
		base.GlobalPosition = mousePosition;
	}
}
