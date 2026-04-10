using System;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Random;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NEpochSlotParticle : Sprite2D
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/epoch_slot_particle_vfx");

	private const double _checkRate = 0.25;

	private double _timer;

	private Control _target;

	private float _speed;

	private const float _deleteDistance = 1500f;

	public static NEpochSlotParticle Create(Control target)
	{
		NEpochSlotParticle nEpochSlotParticle = PreloadManager.Cache.GetScene(scenePath).Instantiate<NEpochSlotParticle>(PackedScene.GenEditState.Disabled);
		nEpochSlotParticle._target = target;
		return nEpochSlotParticle;
	}

	public override void _Ready()
	{
		base.GlobalPosition = _target.GlobalPosition + new Vector2(Rng.Chaotic.NextFloat(40f, 350f) * (Rng.Chaotic.NextBool() ? (-1f) : 1f), Rng.Chaotic.NextFloat(40f, 300f) * (Rng.Chaotic.NextBool() ? (-1f) : 1f));
		float num = Rng.Chaotic.NextFloat(0.5f, 1.5f);
		base.Scale = new Vector2(1f, 1f) * num;
		base.Modulate = new Color(Rng.Chaotic.NextFloat(0.75f, 0.9f), Rng.Chaotic.NextFloat(0.7f, 0.8f), Rng.Chaotic.NextFloat(0f, 0.4f), 0f);
		_speed = Rng.Chaotic.NextFloat(3f, 4f) / num;
		Vector2 vector = _target.GlobalPosition - base.GlobalPosition;
		base.Rotation = Mathf.Atan2(vector.Y, vector.X) + (float)Math.PI / 2f;
		Tween tween = CreateTween();
		tween.TweenProperty(this, "modulate:a", Rng.Chaotic.NextFloat(0.75f, 1f), 0.25);
	}

	public override void _Process(double delta)
	{
		_timer -= delta;
		if (_timer < 0.0)
		{
			_timer += 0.25;
			if (base.GlobalPosition.DistanceSquaredTo(_target.GlobalPosition) < 1500f)
			{
				this.QueueFreeSafely();
			}
		}
		base.GlobalPosition = base.GlobalPosition.Lerp(_target.GlobalPosition, (float)((double)_speed * delta));
	}
}
