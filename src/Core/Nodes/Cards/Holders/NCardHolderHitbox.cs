using Godot;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Cards.Holders;

public partial class NCardHolderHitbox : NButton
{
	protected override string? ClickedSfx => null;

	public override void _Ready()
	{
		ConnectSignals();
	}
}
