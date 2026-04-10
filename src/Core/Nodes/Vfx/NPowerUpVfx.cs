using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NPowerUpVfx : Node2D
{
	private float _timer;

	private const float _vfxDuration = 1f;

	private Control _creatureVisuals;

	private Sprite2D _backVfx;

	private static string NormalScenePath => SceneHelper.GetScenePath("/vfx/vfx_power_up/vfx_power_up");

	private static string GhostlyScenePath => SceneHelper.GetScenePath("/vfx/vfx_ghostly_power_up/vfx_ghostly_power_up");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlyArray<string>(new string[2] { NormalScenePath, GhostlyScenePath });

	public static NPowerUpVfx? CreateNormal(Creature target)
	{
		return CreatePowerUpVfx(target, NormalScenePath);
	}

	public static NPowerUpVfx? CreateGhostly(Creature target)
	{
		return CreatePowerUpVfx(target, GhostlyScenePath);
	}

	private static NPowerUpVfx? CreatePowerUpVfx(Creature target, string scenePath)
	{
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(target);
		if (nCreature == null || !nCreature.IsInteractable)
		{
			return null;
		}
		NPowerUpVfx nPowerUpVfx = PreloadManager.Cache.GetScene(scenePath).Instantiate<NPowerUpVfx>(PackedScene.GenEditState.Disabled);
		nPowerUpVfx.GlobalPosition = nCreature.VfxSpawnPosition;
		NCombatRoom.Instance.CombatVfxContainer.AddChildSafely(nPowerUpVfx);
		return nPowerUpVfx;
	}

	public override void _Ready()
	{
		_timer = 1f;
		_backVfx = GetNode<Sprite2D>("%BackVfx");
		Vector2 globalPosition = _backVfx.GlobalPosition;
		_backVfx.Reparent(NCombatRoom.Instance.BackCombatVfxContainer);
		_backVfx.GlobalPosition = globalPosition;
	}

	public override void _Process(double delta)
	{
		_timer -= (float)delta;
		float a = 1f;
		if (Mathf.Abs(_timer / 1f - 0.5f) > 0.4f)
		{
			a = Mathf.Max(0f, 1f - (Mathf.Abs(_timer / 1f - 0.5f) - 0.4f) / 0.1f);
		}
		base.Modulate = new Color(1f, 1f, 1f, a);
		_backVfx.Modulate = new Color(1f, 1f, 1f, a);
		if (_timer < 0f)
		{
			this.QueueFreeSafely();
			_backVfx.QueueFreeSafely();
		}
	}
}
