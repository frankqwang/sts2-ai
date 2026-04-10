using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx.Cards;

public partial class NExhaustVfx : Node2D
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("vfx/cards/exhaust_vfx");

	private NCard _card;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public static NExhaustVfx? Create(NCard card)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NExhaustVfx nExhaustVfx = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NExhaustVfx>(PackedScene.GenEditState.Disabled);
		nExhaustVfx._card = card;
		return nExhaustVfx;
	}

	public override void _Ready()
	{
		base.GlobalPosition = _card.GlobalPosition;
		foreach (GpuParticles2D item in GetChildren().OfType<GpuParticles2D>())
		{
			item.Emitting = true;
		}
		if (NCombatUi.IsDebugHidingPlayContainer)
		{
			_card.Modulate = Colors.Transparent;
			_card.Visible = false;
			base.Visible = false;
		}
		else
		{
			NDebugAudioManager.Instance?.Play("card_exhaust.mp3");
		}
		TaskHelper.RunSafely(SelfDestruct());
	}

	private async Task SelfDestruct()
	{
		await Task.Delay(2000);
		if (GodotObject.IsInstanceValid(this))
		{
			this.QueueFreeSafely();
		}
	}
}
