using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Random;

namespace MegaCrit.Sts2.Core.Nodes.Screens.FeedbackScreen;

public partial class NSendFeedbackFlower : Control
{
	public enum State
	{
		None,
		Nodding,
		Anticipation,
		NoddingFast
	}

	private const string _normalImage = "res://images/atlases/compressed.sprites/feedback/flower.tres";

	private const string _noddingImage = "res://images/atlases/compressed.sprites/feedback/flower_happy.tres";

	private const string _anticipationImage = "res://images/atlases/compressed.sprites/feedback/flower_anticipation.tres";

	private Tween? _tween;

	private Vector2 _originalPosition;

	public NSendFeedbackCartoon Cartoon { get; private set; }

	public State MyState { get; private set; }

	public override void _Ready()
	{
		_originalPosition = base.Position;
		Cartoon = GetNode<NSendFeedbackCartoon>("Flower");
	}

	public void SetState(State state)
	{
		switch (state)
		{
		case State.Nodding:
			_tween?.Kill();
			_tween = CreateTween();
			_tween.TweenProperty(this, "rotation", Mathf.DegToRad(8f), 0.5);
			_tween.TweenProperty(this, "rotation", Mathf.DegToRad(-8f), 0.5);
			_tween.SetLoops();
			Cartoon.Texture = PreloadManager.Cache.GetTexture2D("res://images/atlases/compressed.sprites/feedback/flower_happy.tres");
			break;
		case State.NoddingFast:
			_tween?.Kill();
			_tween = CreateTween();
			_tween.TweenProperty(this, "rotation", Mathf.DegToRad(8f), 0.2);
			_tween.TweenProperty(this, "rotation", Mathf.DegToRad(-8f), 0.2);
			_tween.SetLoops();
			Cartoon.Texture = PreloadManager.Cache.GetTexture2D("res://images/atlases/compressed.sprites/feedback/flower_happy.tres");
			break;
		case State.Anticipation:
			_tween?.Kill();
			_tween = null;
			Cartoon.Texture = PreloadManager.Cache.GetTexture2D("res://images/atlases/compressed.sprites/feedback/flower_anticipation.tres");
			_tween = CreateTween();
			_tween.TweenInterval(0.05000000074505806);
			_tween.TweenCallback(Callable.From(SetRandomPosition));
			_tween.SetLoops();
			break;
		default:
			base.Rotation = 0f;
			_tween?.Kill();
			_tween = null;
			Cartoon.Texture = PreloadManager.Cache.GetTexture2D("res://images/atlases/compressed.sprites/feedback/flower.tres");
			break;
		}
		MyState = state;
	}

	private void SetRandomPosition()
	{
		base.Position = _originalPosition + new Vector2(Rng.Chaotic.NextFloat(-3f, 3f), Rng.Chaotic.NextFloat(-3f, 3f));
	}
}
