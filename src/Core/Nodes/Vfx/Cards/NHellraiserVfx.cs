using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx.Cards;

public partial class NHellraiserVfx : Control
{
	private float _duration = 1f;

	private int _swordAmount = 10;

	private Vector2 _spawnPosition;

	private static readonly Vector2 _vfxOffset = new Vector2(-100f, -200f);

	private static readonly string _scenePath = SceneHelper.GetScenePath("vfx/cards/vfx_hellraiser/hellraiser_vfx");

	private const string _hellraiserSfxPath = "event:/sfx/characters/ironclad/ironclad_hellraiser";

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public static NHellraiserVfx? Create(Creature target)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NHellraiserVfx nHellraiserVfx = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NHellraiserVfx>(PackedScene.GenEditState.Disabled);
		nHellraiserVfx._spawnPosition = NCombatRoom.Instance.GetCreatureNode(target).GetBottomOfHitbox() + _vfxOffset;
		return nHellraiserVfx;
	}

	public override void _Ready()
	{
		List<float> list = new List<float>();
		for (int i = 0; i < _swordAmount; i++)
		{
			list.Add(Rng.Chaotic.NextFloat(10f, 50f));
		}
		list.Sort();
		foreach (float item in list)
		{
			NHellraiserSwordVfx nHellraiserSwordVfx = NHellraiserSwordVfx.Create();
			nHellraiserSwordVfx.GlobalPosition = _spawnPosition;
			nHellraiserSwordVfx.posY = item;
			float num = MathHelper.Remap(item, 10f, 50f, 0.8f, 1f);
			nHellraiserSwordVfx.targetColor = new Color(num, num, num);
			NCombatRoom.Instance.CombatVfxContainer.AddChildSafely(nHellraiserSwordVfx);
		}
		list = new List<float>();
		for (int j = 0; j < _swordAmount; j++)
		{
			list.Add(Rng.Chaotic.NextFloat(-50f, -10f));
		}
		SfxCmd.Play("event:/sfx/characters/ironclad/ironclad_hellraiser");
		list.Sort();
		foreach (float item2 in list)
		{
			NHellraiserSwordVfx nHellraiserSwordVfx2 = NHellraiserSwordVfx.Create();
			nHellraiserSwordVfx2.GlobalPosition = _spawnPosition;
			nHellraiserSwordVfx2.posY = item2;
			float num2 = MathHelper.Remap(item2, -10f, -50f, 0.7f, 0.4f);
			nHellraiserSwordVfx2.targetColor = new Color(num2, num2, num2);
			NCombatRoom.Instance.BackCombatVfxContainer.AddChildSafely(nHellraiserSwordVfx2);
		}
		NCombatRoom.Instance.CombatVfxContainer.AddChildSafely(NAdditiveOverlayVfx.Create());
		TaskHelper.RunSafely(SelfDestruct());
	}

	private async Task SelfDestruct()
	{
		await Task.Delay(2000);
		this.QueueFreeSafely();
	}
}
