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

public partial class NFanOfKnivesVfx : Node2D
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("vfx/fan_of_knives_vfx");

	private const string _fanOfKnivesSfx = "event:/sfx/characters/silent/silent_fan_of_knives";

	private readonly List<Node2D> _shivs = new List<Node2D>();

	private Node2D _shiv1;

	private Node2D _shiv2;

	private Node2D _shiv3;

	private Node2D _shiv4;

	private Node2D _shiv5;

	private Node2D _shiv6;

	private Node2D _shiv7;

	private Node2D _shiv8;

	private Node2D _shiv9;

	private Vector2 _spawnPosition;

	private const double _fanDuration = 0.8;

	private Tween? _spawnTween;

	private Tween? _fanTween;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public static NFanOfKnivesVfx? Create(Creature target)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NFanOfKnivesVfx nFanOfKnivesVfx = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NFanOfKnivesVfx>(PackedScene.GenEditState.Disabled);
		nFanOfKnivesVfx._spawnPosition = NCombatRoom.Instance.GetCreatureNode(target).VfxSpawnPosition;
		return nFanOfKnivesVfx;
	}

	public override void _Ready()
	{
		_shiv1 = GetNode<Node2D>("ShivFanParticle1");
		_shiv2 = GetNode<Node2D>("ShivFanParticle2");
		_shiv3 = GetNode<Node2D>("ShivFanParticle3");
		_shiv4 = GetNode<Node2D>("ShivFanParticle4");
		_shiv5 = GetNode<Node2D>("ShivFanParticle5");
		_shiv6 = GetNode<Node2D>("ShivFanParticle6");
		_shiv7 = GetNode<Node2D>("ShivFanParticle7");
		_shiv8 = GetNode<Node2D>("ShivFanParticle8");
		_shiv9 = GetNode<Node2D>("ShivFanParticle9");
		_shivs.Add(_shiv1);
		_shivs.Add(_shiv2);
		_shivs.Add(_shiv3);
		_shivs.Add(_shiv4);
		_shivs.Add(_shiv5);
		_shivs.Add(_shiv6);
		_shivs.Add(_shiv7);
		_shivs.Add(_shiv8);
		_shivs.Add(_shiv9);
		foreach (Node2D shiv in _shivs)
		{
			shiv.Scale = Vector2.One * Rng.Chaotic.NextFloat(0.98f, 1.02f);
			shiv.GlobalPosition = _spawnPosition;
		}
		TaskHelper.RunSafely(Animate());
	}

	public override void _ExitTree()
	{
		_fanTween?.Kill();
		_spawnTween?.Kill();
	}

	private async Task Animate()
	{
		SfxCmd.Play("event:/sfx/characters/silent/silent_fan_of_knives");
		_spawnTween = CreateTween().SetParallel();
		foreach (Node2D shiv in _shivs)
		{
			float num = Rng.Chaotic.NextFloat(0.4f, 0.8f);
			_spawnTween.TweenProperty(shiv, "offset:y", -180f, num).From(0f).SetEase(Tween.EaseType.Out)
				.SetTrans(Tween.TransitionType.Back);
			_spawnTween.TweenProperty(shiv, "modulate", Colors.White, num).From(StsColors.transparentBlack);
			_spawnTween.TweenProperty(shiv.GetNode<Node2D>("Shadow"), "offset:y", -180f, num).From(0f).SetEase(Tween.EaseType.Out)
				.SetTrans(Tween.TransitionType.Back);
		}
		_spawnTween.Chain();
		foreach (Node2D shiv2 in _shivs)
		{
			_spawnTween.TweenProperty(shiv2, "modulate", StsColors.transparentWhite, 0.4).SetDelay(Rng.Chaotic.NextDouble(0.25, 0.5));
		}
		_fanTween = CreateTween().SetParallel();
		_fanTween.TweenInterval(0.4000000059604645);
		_fanTween.Chain();
		_fanTween.TweenProperty(_shiv1, "rotation", -1.74533f, 0.8).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back);
		_fanTween.TweenProperty(_shiv2, "rotation", -1.3089975f, 0.8).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back);
		_fanTween.TweenProperty(_shiv3, "rotation", -0.872665f, 0.8).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back);
		_fanTween.TweenProperty(_shiv4, "rotation", -0.4363325f, 0.8).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back);
		_fanTween.TweenProperty(_shiv6, "rotation", 0.4363325f, 0.8).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back);
		_fanTween.TweenProperty(_shiv7, "rotation", 0.872665f, 0.8).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back);
		_fanTween.TweenProperty(_shiv8, "rotation", 1.3089975f, 0.8).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back);
		_fanTween.TweenProperty(_shiv9, "rotation", 1.74533f, 0.8).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back);
		await ToSignal(_spawnTween, Tween.SignalName.Finished);
		this.QueueFreeSafely();
	}
}
