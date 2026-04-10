using Godot;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NExhaustPileButton : NCombatCardPile
{
	private Viewport _viewport;

	private Vector2 _posOffset;

	private static readonly Vector2 _hideOffset = new Vector2(150f, 0f);

	protected override string[] Hotkeys => new string[1] { MegaInput.viewExhaustPileAndTabRight };

	protected override PileType Pile => PileType.Exhaust;

	public override void _Ready()
	{
		ConnectSignals();
		base.Visible = false;
		_viewport = GetViewport();
		_posOffset = new Vector2(base.OffsetRight + 100f, 0f - base.OffsetBottom + 90f);
		GetTree().Root.Connect(Viewport.SignalName.SizeChanged, Callable.From(SetAnimInOutPositions));
		SetAnimInOutPositions();
		Disable();
	}

	public override void Initialize(Player player)
	{
		base.Initialize(player);
		if (Pile.GetPile(player).Cards.Count > 0)
		{
			base.Visible = true;
			base.Position = _showPosition;
			Enable();
		}
	}

	protected override void AddCard()
	{
		base.AddCard();
		if (!base.Visible)
		{
			AnimIn();
		}
		Enable();
	}

	protected override void SetAnimInOutPositions()
	{
		_showPosition = NGame.Instance.Size - _posOffset;
		_hidePosition = _showPosition + _hideOffset;
	}

	public override void AnimIn()
	{
		base.AnimIn();
		base.Visible = true;
	}
}
