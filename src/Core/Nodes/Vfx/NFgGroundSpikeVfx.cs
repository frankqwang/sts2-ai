using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NFgGroundSpikeVfx : NBgGroundSpikeVfx
{
	private const string _scenePath = "res://scenes/vfx/fg_ground_spike_vfx.tscn";

	public new static NFgGroundSpikeVfx? Create(Vector2 position, bool movingRight = true, VfxColor vfxColor = VfxColor.Red)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NFgGroundSpikeVfx nFgGroundSpikeVfx = PreloadManager.Cache.GetScene("res://scenes/vfx/fg_ground_spike_vfx.tscn").Instantiate<NFgGroundSpikeVfx>(PackedScene.GenEditState.Disabled);
		nFgGroundSpikeVfx._startPosition = position;
		nFgGroundSpikeVfx._movingRight = movingRight;
		nFgGroundSpikeVfx._vfxColor = vfxColor;
		return nFgGroundSpikeVfx;
	}

	protected override void AdjustStartPosition()
	{
		_startPosition += new Vector2(_movingRight ? Rng.Chaotic.NextFloat(40f, 160f) : Rng.Chaotic.NextFloat(-160f, -40f), Rng.Chaotic.NextFloat(10f, 32f));
	}
}
