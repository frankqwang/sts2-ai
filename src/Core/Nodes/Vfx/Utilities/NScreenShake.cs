using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;

public partial class NScreenShake : Node
{
	private Control? _shakeTarget;

	private Vector2 _originalTargetPosition;

	private ScreenPunchInstance? _shakeInstance;

	private ScreenRumbleInstance? _rumbleInstance;

	private ScreenTraumaRumble _traumaRumble;

	private float _multiplier;

	private readonly Dictionary<ShakeStrength, float> _strength = new Dictionary<ShakeStrength, float>();

	private readonly Dictionary<ShakeDuration, double> _duration = new Dictionary<ShakeDuration, double>();

	public override void _Ready()
	{
		_traumaRumble = new ScreenTraumaRumble();
		_strength.Add(ShakeStrength.VeryWeak, 2f);
		_strength.Add(ShakeStrength.Weak, 5f);
		_strength.Add(ShakeStrength.Medium, 20f);
		_strength.Add(ShakeStrength.Strong, 40f);
		_strength.Add(ShakeStrength.TooMuch, 80f);
		_duration.Add(ShakeDuration.Short, 0.3);
		_duration.Add(ShakeDuration.Normal, 0.8);
		_duration.Add(ShakeDuration.Long, 1.2);
		_duration.Add(ShakeDuration.Forever, 999999.0);
	}

	public void SetTarget(Control targetScreen)
	{
		_shakeTarget = targetScreen;
		_originalTargetPosition = targetScreen.Position;
	}

	public override void _Process(double delta)
	{
		Vector2 vector = Vector2.Zero;
		if (_rumbleInstance != null)
		{
			vector = _rumbleInstance.Update(delta);
			if (_rumbleInstance.IsDone)
			{
				_rumbleInstance = null;
			}
		}
		if (_shakeInstance != null)
		{
			vector = _shakeInstance.Update(delta);
			if (_shakeInstance.IsDone)
			{
				_shakeInstance = null;
			}
		}
		vector += _traumaRumble.Update(delta);
		if (_shakeTarget != null && _shakeTarget.IsValid())
		{
			_shakeTarget.Position = _originalTargetPosition + vector;
		}
	}

	public void Shake(ShakeStrength strength, ShakeDuration duration, float degAngle)
	{
		if (_shakeTarget == null)
		{
			Log.Error("Missing screenShake target!");
		}
		else
		{
			_shakeInstance = new ScreenPunchInstance(_strength[strength] * _multiplier, _duration[duration], degAngle);
		}
	}

	public void Rumble(ShakeStrength strength, ShakeDuration duration, RumbleStyle style)
	{
		if (_shakeTarget == null)
		{
			Log.Error("Missing screenShake target!");
		}
		else
		{
			_rumbleInstance = new ScreenRumbleInstance(_strength[strength] * _multiplier, _duration[duration], 1f, style);
		}
	}

	public void AddTrauma(ShakeStrength strength)
	{
		_traumaRumble.AddTrauma(strength);
	}

	public void ClearTarget()
	{
		_shakeTarget = null;
		_shakeInstance = null;
		_rumbleInstance = null;
	}

	private void StopRumble()
	{
		_rumbleInstance = null;
	}

	public void SetMultiplier(float multiplier)
	{
		_multiplier = multiplier;
		_traumaRumble.SetMultiplier(multiplier);
	}
}
