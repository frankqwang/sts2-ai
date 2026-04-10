using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NMapCircleVfx : Control
{
	private TextureRect _image;

	private const string _path = "res://scenes/vfx/map_circle_vfx.tscn";

	private static readonly string[] _textures = new string[5] { "res://images/atlases/compressed.sprites/map/map_circle_0.tres", "res://images/atlases/compressed.sprites/map/map_circle_1.tres", "res://images/atlases/compressed.sprites/map/map_circle_2.tres", "res://images/atlases/compressed.sprites/map/map_circle_3.tres", "res://images/atlases/compressed.sprites/map/map_circle_4.tres" };

	private const double _animInterval = 1.0 / 24.0;

	private bool _playAnim;

	public static IEnumerable<string> AssetPaths => _textures.Append("res://scenes/vfx/map_circle_vfx.tscn");

	public override void _Ready()
	{
		_image = GetNode<TextureRect>("TextureRect");
		_image.Texture = PreloadManager.Cache.GetTexture2D(_textures[0]);
		base.RotationDegrees = Rng.Chaotic.NextFloat(360f);
		Vector2 vector = Vector2.One * Rng.Chaotic.NextFloat(0.85f, 0.9f);
		if (_playAnim)
		{
			Tween tween = CreateTween().SetParallel();
			tween.TweenProperty(this, "scale", vector, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
				.From(vector * 2f);
			tween.TweenProperty(this, "modulate:a", 0.95f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
				.From(0f);
			TaskHelper.RunSafely(AnimateSprite());
		}
		else
		{
			base.Scale = vector;
			base.Modulate = new Color(base.Modulate, 0.95f);
			string path = _textures.Last();
			_image.Texture = PreloadManager.Cache.GetTexture2D(path);
		}
	}

	private async Task AnimateSprite()
	{
		string[] textures = _textures;
		foreach (string path in textures)
		{
			_image.Texture = PreloadManager.Cache.GetTexture2D(path);
			SceneTreeTimer source = GetTree().CreateTimer(1.0 / 24.0);
			await ToSignal(source, SceneTreeTimer.SignalName.Timeout);
		}
	}

	public static NMapCircleVfx? Create(bool playAnim)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NMapCircleVfx nMapCircleVfx = PreloadManager.Cache.GetScene("res://scenes/vfx/map_circle_vfx.tscn").Instantiate<NMapCircleVfx>(PackedScene.GenEditState.Disabled);
		nMapCircleVfx._playAnim = playAnim;
		return nMapCircleVfx;
	}
}
