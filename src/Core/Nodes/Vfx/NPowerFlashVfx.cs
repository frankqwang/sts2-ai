using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NPowerFlashVfx : Node2D
{
	private const string _scenePath = "res://scenes/vfx/power_flash_vfx.tscn";

	private Sprite2D _sprite;

	private PowerModel _power;

	private Tween? _spriteTween;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>("res://scenes/vfx/power_flash_vfx.tscn");

	public override void _Ready()
	{
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(_power.Owner);
		if (nCreature == null)
		{
			this.QueueFreeSafely();
			return;
		}
		_sprite = GetNode<Sprite2D>("Sprite2D");
		base.GlobalPosition = nCreature.VfxSpawnPosition;
		TaskHelper.RunSafely(StartVfx());
	}

	public override void _ExitTree()
	{
		_spriteTween?.Kill();
	}

	private async Task StartVfx()
	{
		_sprite.Texture = _power.BigIcon;
		_sprite.Modulate = Colors.White;
		_spriteTween = CreateTween();
		_spriteTween.SetParallel();
		_spriteTween.TweenProperty(_sprite, "scale", Vector2.One * 0.4f, 0.4);
		_spriteTween.TweenProperty(_sprite, "modulate:a", 1, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Sine);
		_spriteTween.SetParallel(parallel: false);
		_spriteTween.TweenProperty(_sprite, "scale", Vector2.One * 0.45f, 0.4);
		_spriteTween.TweenProperty(_sprite, "modulate:a", 0, 0.25).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Sine);
		await ToSignal(_spriteTween, Tween.SignalName.Finished);
		this.QueueFreeSafely();
	}

	public static NPowerFlashVfx? Create(PowerModel power)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		if (!power.ShouldPlayVfx)
		{
			return null;
		}
		NPowerFlashVfx nPowerFlashVfx = (NPowerFlashVfx)PreloadManager.Cache.GetScene("res://scenes/vfx/power_flash_vfx.tscn").Instantiate(PackedScene.GenEditState.Disabled);
		nPowerFlashVfx._power = power;
		return nPowerFlashVfx;
	}
}
