using System;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NBgGroundSpikeVfx : Sprite2D
{
	private const string _scenePath = "res://scenes/vfx/bg_ground_spike_vfx.tscn";

	protected Vector2 _startPosition;

	protected bool _movingRight = true;

	protected VfxColor _vfxColor;

	private Vector2 _velocity;

	private Tween? _tween;

	public static NBgGroundSpikeVfx? Create(Vector2 position, bool movingRight = true, VfxColor vfxColor = VfxColor.Red)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NBgGroundSpikeVfx nBgGroundSpikeVfx = PreloadManager.Cache.GetScene("res://scenes/vfx/bg_ground_spike_vfx.tscn").Instantiate<NBgGroundSpikeVfx>(PackedScene.GenEditState.Disabled);
		nBgGroundSpikeVfx._startPosition = position;
		nBgGroundSpikeVfx._movingRight = movingRight;
		nBgGroundSpikeVfx._vfxColor = vfxColor;
		return nBgGroundSpikeVfx;
	}

	public override void _Ready()
	{
		base.Skew = (_movingRight ? (Rng.Chaotic.NextFloat(15f, 30f) * 0.0174533f) : (Rng.Chaotic.NextFloat(-30f, -15f) * 0.0174533f));
		float num = Rng.Chaotic.NextFloat(0.5f, 1.5f);
		base.Scale = new Vector2(Rng.Chaotic.NextFloat(0.8f, 1.2f), Rng.Chaotic.NextFloat(0.8f, 2f)) * num;
		AdjustStartPosition();
		base.GlobalPosition = _startPosition;
		_velocity = new Vector2(_movingRight ? Rng.Chaotic.NextFloat(50f, 250f) : Rng.Chaotic.NextFloat(-250f, -50f), Rng.Chaotic.NextFloat(-5f, 5f)) / num;
		SetColor();
		TaskHelper.RunSafely(Animate());
	}

	private void SetColor()
	{
		switch (_vfxColor)
		{
		case VfxColor.Red:
			base.Modulate = new Color(1f, Rng.Chaotic.NextFloat(0.2f, 0.8f), Rng.Chaotic.NextFloat(0f, 0.2f), 0.5f);
			break;
		case VfxColor.Purple:
			base.Modulate = new Color(Rng.Chaotic.NextFloat(0f, 0.2f), Rng.Chaotic.NextFloat(0.2f, 0.8f), 1f, 0.5f);
			break;
		case VfxColor.White:
		{
			float num3 = Rng.Chaotic.NextFloat(0.2f, 0.8f);
			base.Modulate = new Color(num3, num3, num3, 0.5f);
			break;
		}
		case VfxColor.Cyan:
		{
			float num2 = Rng.Chaotic.NextFloat(0.6f, 1f);
			base.Modulate = new Color(0.2f, num2, num2, 0.5f);
			break;
		}
		case VfxColor.Gold:
		{
			float num = Rng.Chaotic.NextFloat(0.6f, 1f);
			base.Modulate = new Color(num, num, 0.2f);
			break;
		}
		default:
			Log.Error("Color: " + _vfxColor.ToString() + " not implemented");
			throw new ArgumentOutOfRangeException();
		}
	}

	protected virtual void AdjustStartPosition()
	{
		_startPosition += new Vector2(_movingRight ? Rng.Chaotic.NextFloat(40f, 160f) : Rng.Chaotic.NextFloat(-160f, -40f), Rng.Chaotic.NextFloat(-96f, -10f));
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
	}

	private async Task Animate()
	{
		float num = Rng.Chaotic.NextFloat(0.25f, 1f);
		_tween = CreateTween().SetParallel();
		_tween.TweenInterval(Rng.Chaotic.NextFloat(0.01f, 0.2f));
		_tween.Chain();
		_tween.TweenProperty(this, "skew", _movingRight ? (Rng.Chaotic.NextFloat(30f, 60f) * 0.0174533f) : (Rng.Chaotic.NextFloat(-60f, -30f) * 0.0174533f), num).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(this, "scale", base.Scale * Rng.Chaotic.NextFloat(0.1f, 0.5f), num).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(this, "modulate:a", 0f, num).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		await ToSignal(_tween, Tween.SignalName.Finished);
		this.QueueFreeSafely();
	}

	public override void _Process(double delta)
	{
		base.Position += _velocity * (float)delta;
	}
}
