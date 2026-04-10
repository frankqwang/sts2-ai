using Godot;
using MegaCrit.Sts2.Core.Combat;

namespace MegaCrit.Sts2.Core.Nodes.Vfx.Backgrounds;

public partial class NQueenRepyBgVfx : TextureRect
{
	public override void _EnterTree()
	{
		CombatManager.Instance.CombatSetUp += OnCombatSetUp;
	}

	public override void _ExitTree()
	{
		CombatManager.Instance.CombatSetUp -= OnCombatSetUp;
	}

	private void OnCombatSetUp(CombatState combatState)
	{
		base.Visible = !combatState.RunState.ExtraFields.FreedRepy;
	}
}
