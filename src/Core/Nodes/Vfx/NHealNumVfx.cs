using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Random;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NHealNumVfx : Node2D
{
	private NCreature _creatureNode;

	private string _text;

	private Tween? _tween;

	private Vector2 _velocity;

	private static readonly float _deceleration = 2000f;

	private static readonly Vector2 _positionOffset = new Vector2(0f, -100f);

	private static readonly string _scenePath = SceneHelper.GetScenePath("vfx/vfx_heal_num");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public static NHealNumVfx? Create(Creature target, decimal amount)
	{
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(target);
		if (nCreature == null || !nCreature.IsInteractable)
		{
			return null;
		}
		NHealNumVfx nHealNumVfx = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NHealNumVfx>(PackedScene.GenEditState.Disabled);
		nHealNumVfx._creatureNode = nCreature;
		nHealNumVfx._text = ((int)amount).ToString();
		return nHealNumVfx;
	}

	public override void _Ready()
	{
		MegaLabel node = GetNode<MegaLabel>("Label");
		node.SetTextAutoSize(_text);
		base.GlobalPosition = _creatureNode.VfxSpawnPosition + _positionOffset + new Vector2(Rng.Chaotic.NextFloat(-10f, 10f), Rng.Chaotic.NextFloat(-5f, 5f));
		_velocity = new Vector2(Rng.Chaotic.NextFloat(-100f, 100f), Rng.Chaotic.NextFloat(-600f, -300f));
		node.Scale = Vector2.One * Rng.Chaotic.NextFloat(1.2f, 1.3f);
		base.RotationDegrees = Rng.Chaotic.NextFloat(-5f, 5f);
		TaskHelper.RunSafely(AnimVfx());
	}

	private async Task AnimVfx()
	{
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "modulate:a", 0f, 0.30000001192092896).SetDelay(1.0).SetEase(Tween.EaseType.In)
			.SetTrans(Tween.TransitionType.Quad);
		_tween.TweenProperty(this, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Quad)
			.From(Vector2.One * 2.5f);
		await _tween.ToSignal(_tween, Tween.SignalName.Finished);
		this.QueueFreeSafely();
	}

	public override void _Process(double delta)
	{
		float num = (float)delta;
		base.Position += _velocity * num;
		if (_velocity.LengthSquared() > 1E-06f)
		{
			_velocity -= (_velocity.Normalized() * _deceleration * num).LimitLength(_velocity.Length());
		}
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
	}
}
