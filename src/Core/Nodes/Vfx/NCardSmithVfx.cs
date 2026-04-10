using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NCardSmithVfx : Node2D
{
	private Tween? _tween;

	public const string smithSfx = "card_smith.mp3";

	private bool _willPlaySfx = true;

	private readonly List<CardModel> _cards = new List<CardModel>();

	private NCard? _cardNode;

	private Control _cardContainer;

	private static string ScenePath => SceneHelper.GetScenePath("vfx/vfx_card_smith");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlyArray<string>(new string[2]
	{
		ScenePath,
		TmpSfx.GetPath("card_smith.mp3")
	});

	public float SfxVolume { get; set; } = 1f;

	public static NCardSmithVfx? Create(IEnumerable<CardModel> cards, bool playSfx = true)
	{
		NCardSmithVfx nCardSmithVfx = Create();
		if (nCardSmithVfx == null)
		{
			return null;
		}
		nCardSmithVfx._cards.AddRange(cards);
		nCardSmithVfx._willPlaySfx = playSfx;
		return nCardSmithVfx;
	}

	public static NCardSmithVfx? Create(NCard card, bool playSfx = true)
	{
		NCardSmithVfx nCardSmithVfx = Create();
		if (nCardSmithVfx == null)
		{
			return null;
		}
		nCardSmithVfx._willPlaySfx = playSfx;
		nCardSmithVfx._cardNode = card;
		return nCardSmithVfx;
	}

	public static NCardSmithVfx? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(ScenePath).Instantiate<NCardSmithVfx>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		if (_cardNode != null)
		{
			base.GlobalPosition = _cardNode.GlobalPosition;
			base.GlobalScale = _cardNode.Scale;
			TaskHelper.RunSafely(PlayAnimation());
		}
		else if (_cards.Count > 0)
		{
			TaskHelper.RunSafely(PlayAnimation(_cards));
		}
		else
		{
			TaskHelper.RunSafely(PlayAnimation());
		}
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
	}

	private async Task PlayAnimation()
	{
		if (_willPlaySfx)
		{
			NDebugAudioManager.Instance?.Play("card_smith.mp3", SfxVolume, PitchVariance.Small);
		}
		_tween = CreateTween();
		_tween.Parallel().TweenCallback(Callable.From(delegate
		{
			PlaySubParticles(GetNode<Control>("Spark1"));
		}));
		_tween.TweenInterval(0.25);
		_tween.Chain().TweenCallback(Callable.From(delegate
		{
			PlaySubParticles(GetNode<Control>("Spark2"));
		}));
		_tween.TweenInterval(0.25);
		_tween.Chain().TweenCallback(Callable.From(delegate
		{
			PlaySubParticles(GetNode<Control>("Spark3"));
		}));
		_tween.TweenInterval(0.4000000059604645);
		await ToSignal(_tween, Tween.SignalName.Finished);
		this.QueueFreeSafely();
	}

	private async Task PlayAnimation(IEnumerable<CardModel> cards)
	{
		Control node = GetNode<Control>("%CardContainer");
		List<NCard> cardNodes = new List<NCard>();
		foreach (CardModel card in cards)
		{
			NCard nCard = NCard.Create(card);
			node.AddChildSafely(nCard);
			nCard.UpdateVisuals(PileType.None, CardPreviewMode.Normal);
			cardNodes.Add(nCard);
		}
		_tween = CreateTween();
		foreach (NCard item in cardNodes)
		{
			_tween.Parallel().TweenProperty(item, "scale", Vector2.One * 1f, 0.25).From(Vector2.Zero)
				.SetEase(Tween.EaseType.Out)
				.SetTrans(Tween.TransitionType.Cubic);
		}
		if (_willPlaySfx)
		{
			_tween.Chain().TweenCallback(Callable.From(delegate
			{
				NDebugAudioManager.Instance?.Play("card_smith.mp3", SfxVolume, PitchVariance.Small);
			}));
		}
		_tween.Parallel().TweenCallback(Callable.From(delegate
		{
			PlaySubParticles(GetNode<Control>("Spark1"));
		}));
		_tween.Parallel().TweenCallback(Callable.From(delegate
		{
			NGame.Instance?.ScreenShake(ShakeStrength.Weak, ShakeDuration.Short, 180f + Rng.Chaotic.NextFloat(-10f, 10f));
		}));
		foreach (NCard item2 in cardNodes)
		{
			_tween.Parallel().TweenProperty(item2, "rotation_degrees", 20, 0.05000000074505806).SetTrans(Tween.TransitionType.Elastic)
				.SetEase(Tween.EaseType.Out);
		}
		_tween.TweenInterval(0.25);
		_tween.Chain().TweenCallback(Callable.From(delegate
		{
			PlaySubParticles(GetNode<Control>("Spark2"));
		}));
		foreach (NCard item3 in cardNodes)
		{
			_tween.Parallel().TweenProperty(item3, "rotation_degrees", -10, 0.05000000074505806).SetTrans(Tween.TransitionType.Elastic)
				.SetEase(Tween.EaseType.Out);
		}
		_tween.Parallel().TweenCallback(Callable.From(delegate
		{
			NGame.Instance?.ScreenShake(ShakeStrength.Weak, ShakeDuration.Short, 180f + Rng.Chaotic.NextFloat(-10f, 10f));
		}));
		_tween.TweenInterval(0.25);
		_tween.Chain().TweenCallback(Callable.From(delegate
		{
			PlaySubParticles(GetNode<Control>("Spark3"));
		}));
		foreach (NCard item4 in cardNodes)
		{
			_tween.Parallel().TweenProperty(item4, "rotation_degrees", 5, 0.05000000074505806).SetTrans(Tween.TransitionType.Elastic)
				.SetEase(Tween.EaseType.Out);
		}
		_tween.Parallel().TweenCallback(Callable.From(delegate
		{
			NGame.Instance?.ScreenShake(ShakeStrength.Weak, ShakeDuration.Short, 180f + Rng.Chaotic.NextFloat(-10f, 10f));
		}));
		_tween.TweenInterval(0.4000000059604645);
		await ToSignal(_tween, Tween.SignalName.Finished);
		if (cardNodes[0].IsInsideTree() && _cards[0].Pile == null)
		{
			_tween = CreateTween();
			foreach (NCard item5 in cardNodes)
			{
				_tween.SetParallel().TweenProperty(item5, "scale", Vector2.Zero, 0.15000000596046448);
			}
			await ToSignal(_tween, Tween.SignalName.Finished);
		}
		else
		{
			if (!cardNodes[0].IsInsideTree())
			{
				return;
			}
			for (int i = 0; i < cardNodes.Count; i++)
			{
				Vector2 targetPosition = cardNodes[i].Model.Pile.Type.GetTargetPosition(cardNodes[i]);
				Vector2 globalPosition = cardNodes[i].GlobalPosition;
				cardNodes[i].Reparent(this);
				cardNodes[i].GlobalPosition = globalPosition;
				NCardFlyVfx nCardFlyVfx = NCardFlyVfx.Create(cardNodes[i], targetPosition, isAddingToPile: false, cardNodes[i].Model.Owner.Character.TrailPath);
				NRun.Instance?.GlobalUi.TopBar.TrailContainer.AddChildSafely(nCardFlyVfx);
				if (nCardFlyVfx.SwooshAwayCompletion != null && i == cardNodes.Count - 1)
				{
					await nCardFlyVfx.SwooshAwayCompletion.Task;
				}
			}
			this.QueueFreeSafely();
		}
	}

	private void PlaySubParticles(Node node)
	{
		foreach (CpuParticles2D item in node.GetChildren().OfType<CpuParticles2D>())
		{
			item.Emitting = true;
		}
	}
}
