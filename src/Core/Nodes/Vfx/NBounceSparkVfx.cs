using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NBounceSparkVfx : Node2D
{
	private const string _scenePath = "res://scenes/vfx/bounce_spark_vfx.tscn";

	[Export(PropertyHint.None, "")]
	private Node2D _particle;

	private Vector2 _velocity;

	private Vector2 _startPosition;

	private float _floorY;

	private const float _targetAlpha = 0.8f;

	private static readonly Vector2 _gravity = new Vector2(0f, 1500f);

	private Tween? _tween;

	public static NBounceSparkVfx? Create(Creature target, VfxColor vfxColor = VfxColor.Gold)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NBounceSparkVfx nBounceSparkVfx = PreloadManager.Cache.GetScene("res://scenes/vfx/bounce_spark_vfx.tscn").Instantiate<NBounceSparkVfx>(PackedScene.GenEditState.Disabled);
		nBounceSparkVfx._startPosition = NCombatRoom.Instance.GetCreatureNode(target).GetBottomOfHitbox();
		return nBounceSparkVfx;
	}

	public override void _Ready()
	{
		_startPosition += new Vector2(Rng.Chaotic.NextFloat(-120f, 120f), Rng.Chaotic.NextFloat(0f, 20f));
		_floorY = _startPosition.Y + Rng.Chaotic.NextFloat(0f, 64f);
		base.GlobalPosition = _startPosition;
		_velocity = new Vector2(Rng.Chaotic.NextFloat(-400f, 400f), Rng.Chaotic.NextFloat(-800f, -300f));
		float num = Rng.Chaotic.NextFloat(0.8f, 1.2f);
		base.Scale = new Vector2(num, 2f - num) * Rng.Chaotic.NextFloat(0.1f, 0.8f);
		base.Modulate = new Color(1f, Rng.Chaotic.NextFloat(0.2f, 0.8f), Rng.Chaotic.NextFloat(0f, 0.2f), 0f);
		TaskHelper.RunSafely(Animate());
	}

	public override void _Process(double delta)
	{
		float num = (float)delta;
		base.Position += _velocity * num;
		_velocity += _gravity * num;
		base.Rotation = MathHelper.GetAngle(_velocity);
		if (base.Position.Y > _floorY)
		{
			base.Position = new Vector2(base.Position.X, _floorY);
			_velocity = new Vector2(_velocity.X, (0f - _velocity.Y) * 0.5f);
		}
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
	}

	private async Task Animate()
	{
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "modulate:a", 0.8f, 0.10000000149011612);
		_tween.Chain();
		float num = Rng.Chaotic.NextFloat(0.4f, 1.5f);
		_tween.TweenProperty(this, "modulate:a", 0f, num);
		_tween.TweenProperty(this, "scale", base.Scale * 0.5f, num);
		await ToSignal(_tween, Tween.SignalName.Finished);
		this.QueueFreeSafely();
	}
}
