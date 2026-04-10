using Godot;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Acts;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Random;

namespace MegaCrit.Sts2.Core.Debug;

public partial class DebugActMap : Node
{
	private ActMap _testActMap;

	public override void _Ready()
	{
		GenerateMap();
	}

	public override void _Process(double delta)
	{
		if (Input.IsKeyPressed(Key.Bracketleft))
		{
			GenerateMap();
		}
	}

	private void GenerateMap()
	{
		uint seed = Rng.Chaotic.NextUnsignedInt();
		_testActMap = new StandardActMap(new Rng(seed), ModelDb.Act<Overgrowth>(), isMultiplayer: false, shouldReplaceTreasureWithElites: false);
		GetNode<NMapScreen>("%MapScreen").SetMap(_testActMap, seed, clearDrawings: true);
	}
}
