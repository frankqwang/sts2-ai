using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx.Cards;

public partial class NBolasVfx : Node2D
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("vfx/cards/bolas_vfx");

	private Node2D _bola2;

	private Node2D _bola3;

	private Vector2 _startPosition;

	private Vector2 _controlPosition;

	private Vector2 _endPosition;

	private float _rotationSpeed = 30f;

	private Tween? _tween;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public static NBolasVfx? Create(Creature owner, Creature target)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		if (target.IsDead)
		{
			return null;
		}
		NBolasVfx nBolasVfx = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NBolasVfx>(PackedScene.GenEditState.Disabled);
		nBolasVfx.GlobalPosition = NCombatRoom.Instance.GetCreatureNode(owner).VfxSpawnPosition;
		nBolasVfx._startPosition = nBolasVfx.GlobalPosition;
		nBolasVfx._endPosition = NCombatRoom.Instance.GetCreatureNode(target).VfxSpawnPosition;
		float y = Mathf.Min(nBolasVfx._startPosition.Y, nBolasVfx._endPosition.Y) - Rng.Chaotic.NextFloat(400f, 500f);
		nBolasVfx._controlPosition = new Vector2((nBolasVfx._startPosition.X + nBolasVfx._endPosition.X) * 0.5f, y);
		return nBolasVfx;
	}

	public override void _Ready()
	{
		TaskHelper.RunSafely(FlyBolasFly());
		_bola2 = GetNode<Node2D>("Bola2");
		_bola3 = GetNode<Node2D>("Bola3");
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
	}

	private async Task FlyBolasFly()
	{
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "modulate:a", 1f, 0.25).From(0f);
		_tween.TweenMethod(Callable.From<float>(FollowCurve), 0f, 1f, 0.6000000238418579).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Sine);
		_tween.Chain();
		_tween.TweenInterval(0.15000000596046448);
		_tween.TweenProperty(this, "modulate:a", 0f, 0.15000000596046448);
		await ToSignal(_tween, Tween.SignalName.Finished);
		this.QueueFreeSafely();
	}

	private void FollowCurve(float progressPercent)
	{
		base.GlobalPosition = MathHelper.BezierCurve(_startPosition, _endPosition, _controlPosition, progressPercent);
	}

	public override void _Process(double delta)
	{
		float num = (float)delta;
		base.Rotation += num * (0f - _rotationSpeed);
		_rotationSpeed -= num * 12f;
		_bola2.Position -= new Vector2(150f * num, 0f);
		_bola3.Position -= new Vector2(-150f * num, 0f);
	}
}
