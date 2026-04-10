using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NRelicFlashVfx : Control
{
	public const float activationDuration = 1f;

	private const string _scenePath = "res://scenes/vfx/relic_flash_vfx.tscn";

	private TextureRect _sprite;

	private TextureRect _sprite2;

	private TextureRect _sprite3;

	private static readonly Vector2 _targetScale = Vector2.One * 1.25f;

	private RelicModel? _relic;

	private Creature? _target;

	private Tween? _tween;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>("res://scenes/vfx/relic_flash_vfx.tscn");

	public static NRelicFlashVfx? Create(RelicModel relic)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NRelicFlashVfx nRelicFlashVfx = PreloadManager.Cache.GetScene("res://scenes/vfx/relic_flash_vfx.tscn").Instantiate<NRelicFlashVfx>(PackedScene.GenEditState.Disabled);
		nRelicFlashVfx._relic = relic;
		return nRelicFlashVfx;
	}

	public static NRelicFlashVfx? Create(RelicModel relic, Creature target)
	{
		NRelicFlashVfx nRelicFlashVfx = Create(relic);
		if (nRelicFlashVfx == null)
		{
			return null;
		}
		nRelicFlashVfx._target = target;
		return nRelicFlashVfx;
	}

	public override void _Ready()
	{
		_sprite = GetNode<TextureRect>("Image1");
		_sprite2 = GetNode<TextureRect>("Image2");
		_sprite3 = GetNode<TextureRect>("Image3");
		if (_target != null)
		{
			base.GlobalPosition = NCombatRoom.Instance.GetCreatureNode(_target).GetTopOfHitbox();
		}
		TaskHelper.RunSafely(StartVfx());
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
	}

	private async Task StartVfx()
	{
		_sprite.Texture = _relic.Icon;
		_sprite2.Texture = _relic.Icon;
		_sprite3.Texture = _relic.Icon;
		_tween = CreateTween().SetParallel();
		if (_target != null)
		{
			base.Position += new Vector2(0f, 64f);
			_tween.TweenProperty(this, "position:y", base.Position.Y - 64f, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		}
		_tween.TweenProperty(_sprite, "modulate:a", 1f, 0.01);
		_tween.TweenProperty(_sprite, "scale", _targetScale, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		_tween.TweenProperty(_sprite, "modulate:a", 0f, 1.5).SetDelay(0.01);
		_tween.TweenProperty(_sprite2, "modulate:a", 1f, 0.01).SetDelay(0.2);
		_tween.TweenProperty(_sprite2, "scale", _targetScale, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic)
			.SetDelay(0.2);
		_tween.TweenProperty(_sprite2, "modulate:a", 0f, 1.5).SetDelay(0.21);
		_tween.TweenProperty(_sprite3, "modulate:a", 1f, 0.01).SetDelay(0.4);
		_tween.TweenProperty(_sprite3, "scale", _targetScale, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic)
			.SetDelay(0.4);
		_tween.TweenProperty(_sprite3, "modulate:a", 0f, 1.5).SetDelay(0.41);
		await ToSignal(_tween, Tween.SignalName.Finished);
		this.QueueFreeSafely();
	}
}
