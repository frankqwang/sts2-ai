using Godot;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NCombatPilesContainer : Control
{
	public static readonly string scenePath = SceneHelper.GetScenePath("combat/combat_piles_container");

	private NDrawPileButton _drawPile;

	private NDiscardPileButton _discardPile;

	private NExhaustPileButton _exhaustPile;

	public NDrawPileButton DrawPile => _drawPile;

	public NDiscardPileButton DiscardPile => _discardPile;

	public NExhaustPileButton ExhaustPile => _exhaustPile;

	public override void _Ready()
	{
		_drawPile = GetNode<NDrawPileButton>("%DrawPile");
		_discardPile = GetNode<NDiscardPileButton>("%DiscardPile");
		_exhaustPile = GetNode<NExhaustPileButton>("%ExhaustPile");
	}

	public void Initialize(Player player)
	{
		_drawPile.Initialize(player);
		_discardPile.Initialize(player);
		_exhaustPile.Initialize(player);
	}

	public void AnimIn()
	{
		_drawPile.AnimIn();
		_discardPile.AnimIn();
	}

	public void AnimOut()
	{
		_drawPile.AnimOut();
		_discardPile.AnimOut();
		_exhaustPile.AnimOut();
	}

	public void Enable()
	{
		_drawPile.Enable();
		_discardPile.Enable();
		_exhaustPile.Enable();
	}

	public void Disable()
	{
		_drawPile.Disable();
		_discardPile.Disable();
		_exhaustPile.Disable();
	}
}
