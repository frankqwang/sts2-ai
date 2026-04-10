using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx.Cards;

public partial class NHellraiserSwordVfx : Control
{
	private static readonly StringName _swordStr = new StringName("Sword");

	private static readonly string _scenePath = SceneHelper.GetScenePath("vfx/cards/vfx_hellraiser/hellraiser_sword_vfx");

	private TextureRect _sword;

	public float posY;

	public Color targetColor;

	public static NHellraiserSwordVfx? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NHellraiserSwordVfx>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		base.Name = _swordStr;
		_sword = GetNode<TextureRect>("TextureRect");
		_sword.FlipH = Rng.Chaotic.NextBool();
		_sword.RotationDegrees = Rng.Chaotic.NextFloat(-20f, 20f);
		_sword.Position = new Vector2(25f, 200f);
		base.Scale = new Vector2(Rng.Chaotic.NextFloat(0.7f, 0.9f), Rng.Chaotic.NextFloat(0.8f, 1.2f)) * Rng.Chaotic.NextFloat(1f, 2f);
		base.Position += new Vector2(Rng.Chaotic.NextGaussianFloat(0f, 1f, -500f, 500f), posY);
		Tween tween = CreateTween().SetParallel();
		tween.TweenInterval(Rng.Chaotic.NextDouble() * 0.8);
		tween.Chain();
		tween.TweenProperty(_sword, "position:y", 80f, 0.25).From(300f).SetEase(Tween.EaseType.Out)
			.SetTrans(Tween.TransitionType.Spring);
		tween.TweenProperty(this, "modulate", targetColor, 0.25).From(Colors.Red);
		tween.Chain().TweenInterval(0.25);
		tween.Chain();
		tween.TweenProperty(_sword, "position:y", 200f, 0.5).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Expo);
		tween.TweenProperty(this, "modulate:a", 0f, 0.5);
		tween.Chain().TweenCallback(Callable.From(OnTweenFinished));
	}

	private void OnTweenFinished()
	{
		this.QueueFreeSafely();
	}
}
