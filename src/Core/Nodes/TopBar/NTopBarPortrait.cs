using Godot;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.sts2.Core.Nodes.TopBar;

public partial class NTopBarPortrait : Control
{
	public void Initialize(Player player)
	{
		this.AddChildSafely(player.Character.Icon);
	}
}
