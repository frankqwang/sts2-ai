using System;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;

namespace MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;

public partial class NLogoAnimation : Control, IScreenContext
{
	private const string _scenePath = "res://scenes/screens/main_menu/logo_animation.tscn";

	private Control _bg;

	private Control _logoContainer;

	private Node2D _logoSpineNode;

	private MegaSprite _spineSprite;

	private Color _logoBgColor = new Color("074254FF");

	private Tween? _tween;

	private bool _cancelled;

	public static string[] AssetPaths => new string[1] { "res://scenes/screens/main_menu/logo_animation.tscn" };

	public Control? DefaultFocusedControl => null;

	public static NLogoAnimation Create()
	{
		return PreloadManager.Cache.GetScene("res://scenes/screens/main_menu/logo_animation.tscn").Instantiate<NLogoAnimation>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		_bg = GetNode<Control>("%Bg");
		_logoContainer = GetNode<Control>("%Container");
		_logoSpineNode = GetNode<Node2D>("Container/SpineSprite");
		_spineSprite = new MegaSprite(_logoSpineNode);
		_logoSpineNode.Visible = false;
		Rect2 bounds = _spineSprite.GetSkeleton().GetBounds();
		float num = Math.Min(base.Size.X * 0.33f / bounds.Size.X, base.Size.Y * 0.33f / bounds.Size.Y);
		_logoSpineNode.Scale = num * Vector2.One;
		_logoSpineNode.Position = -bounds.Size * _logoSpineNode.Scale * 0.5f;
	}

	public async Task PlayAnimation(CancellationToken token)
	{
		if (token.IsCancellationRequested)
		{
			_cancelled = true;
			return;
		}
		_tween = CreateTween();
		_tween.TweenInterval(1.0);
		await ToSignal(_tween, Tween.SignalName.Finished);
		if (token.IsCancellationRequested)
		{
			_cancelled = true;
			return;
		}
		_logoSpineNode.Visible = true;
		_spineSprite.GetAnimationState().SetAnimation("animation", loop: false);
		NDebugAudioManager.Instance.Play("SOTE_Logo_Echoing_ShortTail.mp3");
		_tween.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_logoSpineNode, "position:y", _logoSpineNode.Position.Y, 0.5).From(_logoSpineNode.Position.Y - 800f).SetEase(Tween.EaseType.Out)
			.SetTrans(Tween.TransitionType.Back);
		_tween.TweenProperty(_logoContainer, "modulate", Colors.White, 0.5);
		await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		while (!_spineSprite.GetAnimationState().GetCurrent(0).IsComplete())
		{
			if (token.IsCancellationRequested)
			{
				_cancelled = true;
				_tween.Kill();
				_tween = CreateTween().SetParallel();
				_tween.TweenProperty(_logoContainer, "modulate", StsColors.transparentWhite, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
				await ToSignal(_tween, Tween.SignalName.Finished);
				break;
			}
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		}
		if (!_cancelled)
		{
			_tween.Kill();
			_tween = CreateTween();
			_tween.TweenProperty(_bg, "modulate", _logoBgColor, 2.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
			_tween.Chain();
			_tween.TweenInterval(1.0);
			await ToSignal(_tween, Tween.SignalName.Finished);
		}
	}
}
