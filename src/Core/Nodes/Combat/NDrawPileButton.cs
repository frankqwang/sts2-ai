using Godot;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Localization;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NDrawPileButton : NCombatCardPile
{
	protected override string[] Hotkeys => new string[1] { MegaInput.viewDrawPile };

	protected override PileType Pile => PileType.Draw;

	public override void _Ready()
	{
		ConnectSignals();
		_emptyPileMessage = new LocString("combat_messages", "OPEN_EMPTY_DRAW");
	}

	protected override void SetAnimInOutPositions()
	{
		_showPosition = base.Position;
		_hidePosition = base.Position + new Vector2(-150f, 100f);
	}
}
