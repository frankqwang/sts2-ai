using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models.Monsters;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NDamageNumVfx : Node2D
{
	private Vector2 _globalSpawnPosition;

	private string _text;

	private Tween? _tween;

	private Vector2 _velocity;

	private static readonly Vector2 _gravity = new Vector2(0f, 2000f);

	private static readonly Vector2 _positionOffset = new Vector2(0f, -100f);

	private static readonly string _scenePath = SceneHelper.GetScenePath("vfx/vfx_damage_num");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public static NDamageNumVfx? Create(Creature target, DamageResult result)
	{
		int num = result.UnblockedDamage;
		if (!(target.Monster is Osty))
		{
			num += result.OverkillDamage;
		}
		return Create(target, num);
	}

	public static NDamageNumVfx? Create(Creature target, int damage, bool requireInteractable = true)
	{
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(target);
		Vector2 globalPosition = Vector2.Zero;
		if (requireInteractable && (nCreature == null || !nCreature.IsInteractable))
		{
			if (!LocalContext.IsMe(target))
			{
				return null;
			}
			Vector2 size = ((SceneTree)Engine.GetMainLoop()).Root.GetViewport().GetVisibleRect().Size;
			globalPosition = size * new Vector2(0.25f, 0.5f);
		}
		else if (nCreature != null)
		{
			globalPosition = nCreature.VfxSpawnPosition + _positionOffset + new Vector2(Rng.Chaotic.NextFloat(-10f, 10f), Rng.Chaotic.NextFloat(-5f, 5f));
		}
		return Create(globalPosition, damage);
	}

	public static NDamageNumVfx? Create(Vector2 globalPosition, int damage)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NDamageNumVfx nDamageNumVfx = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NDamageNumVfx>(PackedScene.GenEditState.Disabled);
		nDamageNumVfx._globalSpawnPosition = globalPosition;
		nDamageNumVfx._text = damage.ToString();
		return nDamageNumVfx;
	}

	public override void _Ready()
	{
		MegaLabel node = GetNode<MegaLabel>("Label");
		node.SetTextAutoSize(_text);
		base.GlobalPosition = _globalSpawnPosition;
		_velocity = new Vector2(Rng.Chaotic.NextFloat(-100f, 100f), Rng.Chaotic.NextFloat(-800f, -700f));
		node.Scale = Vector2.One * Rng.Chaotic.NextFloat(1.2f, 1.3f);
		base.RotationDegrees = Rng.Chaotic.NextFloat(-5f, 5f);
		TaskHelper.RunSafely(AnimVfx());
	}

	private async Task AnimVfx()
	{
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "modulate", StsColors.cream, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		_tween.TweenProperty(this, "modulate:a", 0f, 2.0).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Quad);
		_tween.TweenProperty(this, "scale", Vector2.One, 1.2000000476837158).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Quad)
			.From(Vector2.One * 2.5f);
		await _tween.ToSignal(_tween, Tween.SignalName.Finished);
		this.QueueFreeSafely();
	}

	public override void _Process(double delta)
	{
		float num = (float)delta;
		base.Position += _velocity * num;
		_velocity += _gravity * num;
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
	}
}
