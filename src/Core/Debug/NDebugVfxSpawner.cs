using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Relics;
using MegaCrit.Sts2.Core.Nodes.Vfx;

namespace MegaCrit.Sts2.Core.Debug;

public partial class NDebugVfxSpawner : Control
{
	private static readonly StringName _viewDeck = new StringName("view_deck");

	private void Spawn()
	{
		NRelicFlashVfx nRelicFlashVfx = NRelicFlashVfx.Create(ModelDb.Relic<PaelsEye>());
		if (nRelicFlashVfx != null)
		{
			this.AddChildSafely(nRelicFlashVfx);
			nRelicFlashVfx.Scale = Vector2.One * 2f;
			nRelicFlashVfx.Position = Size * 0.5f;
		}
	}

	public override void _Process(double delta)
	{
		if (Input.IsActionJustReleased(_viewDeck))
		{
			Spawn();
		}
	}
}
