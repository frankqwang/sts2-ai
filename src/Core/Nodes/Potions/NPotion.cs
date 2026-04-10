using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Vfx;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Potions;

public partial class NPotion : Control
{
	private const float _newlyAcquiredPopDuration = 0.35f;

	private const float _newlyAcquiredFadeInDuration = 0.1f;

	private const float _newlyAcquiredPopDistance = 40f;

	private PotionModel? _model;

	private Control _container;

	private Tween? _bounceTween;

	private Tween? _obtainedTween;

	private CancellationTokenSource? _cancellationTokenSource;

	public TextureRect Image { get; private set; }

	public TextureRect Outline { get; private set; }

	private static string ScenePath => SceneHelper.GetScenePath("/potions/potion");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlyArray<string>(new string[2]
	{
		ScenePath,
		NPotionFlashVfx.ScenePath
	});

	public PotionModel Model
	{
		get
		{
			return _model ?? throw new InvalidOperationException("Model was accessed before it was set.");
		}
		set
		{
			value.AssertMutable();
			_model = value;
			Reload();
		}
	}

	public static NPotion? Create(PotionModel potion)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NPotion nPotion = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NPotion>(PackedScene.GenEditState.Disabled);
		nPotion.Model = potion;
		return nPotion;
	}

	public override void _Ready()
	{
		Image = GetNode<TextureRect>("%Image");
		Outline = GetNode<TextureRect>("%Outline");
		_container = GetNode<Control>("Container");
		Reload();
	}

	private void Reload()
	{
		if (IsNodeReady() && _model != null)
		{
			Image.Texture = _model.Image;
			Outline.Texture = _model.Outline;
		}
	}

	public async Task PlayNewlyAcquiredAnimation(Vector2? startLocation)
	{
		if (_cancellationTokenSource != null)
		{
			await _cancellationTokenSource.CancelAsync();
		}
		CancellationTokenSource cancelTokenSource = (_cancellationTokenSource = new CancellationTokenSource());
		await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		if (!cancelTokenSource.IsCancellationRequested)
		{
			_obtainedTween?.Kill();
			if (!startLocation.HasValue)
			{
				Control container = _container;
				Vector2 position = _container.Position;
				position.Y = 40f;
				container.Position = position;
				Control container2 = _container;
				Color modulate = _container.Modulate;
				modulate.A = 0f;
				container2.Modulate = modulate;
				_obtainedTween = GetTree().CreateTween();
				_obtainedTween.TweenProperty(_container, "modulate:a", 1f, 0.10000000149011612);
				_obtainedTween.Parallel();
				_obtainedTween.SetEase(Tween.EaseType.Out);
				_obtainedTween.SetTrans(Tween.TransitionType.Back);
				_obtainedTween.TweenProperty(_container, "position:y", 0f, 0.3499999940395355);
				_obtainedTween.TweenCallback(Callable.From(DoFlash));
			}
			else
			{
				_container.GlobalPosition = startLocation.Value;
				Control container3 = _container;
				Color modulate = _container.Modulate;
				modulate.A = 1f;
				container3.Modulate = modulate;
				_obtainedTween = GetTree().CreateTween();
				_obtainedTween.SetEase(Tween.EaseType.Out);
				_obtainedTween.SetTrans(Tween.TransitionType.Quad);
				_obtainedTween.TweenProperty(_container, "position", Vector2.Zero, 0.3499999940395355);
				_obtainedTween.TweenCallback(Callable.From(DoFlash));
			}
		}
	}

	private void DoFlash()
	{
		this.AddChildSafely(NPotionFlashVfx.Create(this));
	}

	public void DoBounce()
	{
		_bounceTween?.Kill();
		_bounceTween = CreateTween();
		_bounceTween.TweenProperty(_container, "position:y", base.Position.Y - 12f, 0.125).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Sine);
		_bounceTween.TweenProperty(_container, "position:y", 0f, 0.125).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Sine);
	}
}
