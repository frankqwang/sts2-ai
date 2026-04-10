using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Potions;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NPotionFlashVfx : Node2D
{
	private Control _flash;

	public static string ScenePath => SceneHelper.GetScenePath("vfx/vfx_potion_flash");

	public static NPotionFlashVfx? Create(NPotion originPotion)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NPotionFlashVfx nPotionFlashVfx = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NPotionFlashVfx>(PackedScene.GenEditState.Disabled);
		nPotionFlashVfx._flash = (Control)originPotion.Duplicate();
		ulong instanceId = nPotionFlashVfx._flash.GetInstanceId();
		nPotionFlashVfx._flash.SetScript(default(Variant));
		nPotionFlashVfx._flash = (Control)GodotObject.InstanceFromId(instanceId);
		return nPotionFlashVfx;
	}

	public override void _Ready()
	{
		GetNode<SubViewport>("SubViewport").AddChildSafely(_flash);
		_flash.Position = Vector2.Zero;
		TaskHelper.RunSafely(FlashAndFree());
	}

	private async Task FlashAndFree()
	{
		CpuParticles2D node = GetNode<CpuParticles2D>("%Flash");
		node.Emitting = true;
		await ToSignal(node, CpuParticles2D.SignalName.Finished);
		this.QueueFreeSafely();
	}
}
