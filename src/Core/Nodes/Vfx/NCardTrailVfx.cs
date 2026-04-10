using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NCardTrailVfx : Node2D
{
	private Control _nodeToFollow;

	private Node2D _sprites;

	private bool _updateSprites = true;

	private Tween? _tween;

	public static NCardTrailVfx? Create(Control card, string characterTrailPath)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCardTrailVfx nCardTrailVfx = PreloadManager.Cache.GetScene(characterTrailPath).Instantiate<NCardTrailVfx>(PackedScene.GenEditState.Disabled);
		nCardTrailVfx._nodeToFollow = card;
		return nCardTrailVfx;
	}

	public override void _Ready()
	{
		if (NCombatUi.IsDebugHidingPlayContainer)
		{
			base.Visible = false;
		}
		_sprites = GetNode<Node2D>("Sprites");
		_sprites.Modulate = StsColors.transparentWhite;
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_sprites, "scale", Vector2.One * 0.5f, 0.5).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Cubic)
			.SetDelay(0.25);
		_tween.TweenProperty(_nodeToFollow, "modulate:a", 0.75f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		_tween.TweenProperty(_sprites, "modulate:a", 1f, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
	}

	public override void _Process(double delta)
	{
		if (_updateSprites)
		{
			base.GlobalPosition = _nodeToFollow.GlobalPosition;
			base.Rotation = _nodeToFollow.Rotation;
		}
	}

	public async Task FadeOut()
	{
		_updateSprites = false;
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "modulate:a", 0f, 0.5);
		StopParticles(_tween);
		await ToSignal(_tween, Tween.SignalName.Finished);
		this.QueueFreeSafely();
	}

	private void StopParticles(Tween tween)
	{
		foreach (Node child in _sprites.GetChildren())
		{
			if (child is CpuParticles2D cpuParticles2D)
			{
				tween.TweenProperty(cpuParticles2D, "amount", 1, 0.5);
			}
		}
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
	}
}
