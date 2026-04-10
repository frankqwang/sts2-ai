using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NUiFlashVfx : Control
{
	private const string _scenePath = "res://scenes/vfx/ui_flash_vfx.tscn";

	private TextureRect _textureRect;

	private Texture2D _texture;

	private Color _modulate;

	private Tween? _spriteTween;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>("res://scenes/vfx/ui_flash_vfx.tscn");

	public override void _Ready()
	{
		_textureRect = GetNode<TextureRect>("TextureRect");
		_textureRect.Texture = _texture;
	}

	public async Task StartVfx()
	{
		TextureRect textureRect = _textureRect;
		Color modulate = _modulate;
		modulate.A = 0f;
		textureRect.Modulate = modulate;
		_textureRect.PivotOffset = _textureRect.Size * 0.5f;
		_spriteTween = CreateTween();
		_spriteTween.SetParallel();
		_spriteTween.TweenProperty(_textureRect, "scale", Vector2.One * 1.3f, 0.5);
		_spriteTween.TweenProperty(_textureRect, "modulate:a", 1f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Sine);
		_spriteTween.TweenProperty(_textureRect, "modulate:a", 0f, 0.25).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Sine)
			.SetDelay(0.3499999940395355);
		await ToSignal(_spriteTween, Tween.SignalName.Finished);
		this.QueueFreeSafely();
	}

	public static NUiFlashVfx? Create(Texture2D tex, Color modulate)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NUiFlashVfx nUiFlashVfx = (NUiFlashVfx)PreloadManager.Cache.GetScene("res://scenes/vfx/ui_flash_vfx.tscn").Instantiate(PackedScene.GenEditState.Disabled);
		nUiFlashVfx._texture = tex;
		nUiFlashVfx._modulate = modulate;
		return nUiFlashVfx;
	}

	public override void _ExitTree()
	{
		_spriteTween?.Kill();
	}
}
