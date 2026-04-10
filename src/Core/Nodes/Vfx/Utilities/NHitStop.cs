using System;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;

public partial class NHitStop : Node
{
	private const float _minTimeScale = 0.1f;

	private CancellationTokenSource? _cancelToken;

	public void DoHitStop(ShakeStrength strength, ShakeDuration duration)
	{
		_cancelToken?.Cancel();
		_cancelToken = new CancellationTokenSource();
		TaskHelper.RunSafely(HitStopTask(EaseForStrength(strength), SecondsForDuration(duration)));
	}

	private async Task HitStopTask(Ease.Functions easing, float seconds)
	{
		SetTimeScale(0.1f);
		ulong lastTicks = Time.GetTicksMsec();
		float timer = 0f;
		while (timer <= seconds)
		{
			await GetTree().ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
			if (_cancelToken?.IsCancellationRequested ?? false)
			{
				return;
			}
			timer += (float)(Time.GetTicksMsec() - lastTicks) / 1000f;
			float num = Ease.Interpolate(timer / seconds, easing);
			float timeScale = Mathf.Min(0.1f + num * 0.9f, 1f);
			SetTimeScale(timeScale);
			lastTicks = Time.GetTicksMsec();
		}
		SetTimeScale(1f);
	}

	private void SetTimeScale(float timeScale)
	{
		Engine.SetTimeScale(timeScale);
	}

	private Ease.Functions EaseForStrength(ShakeStrength strength)
	{
		return strength switch
		{
			ShakeStrength.VeryWeak => Ease.Functions.CircIn, 
			ShakeStrength.Weak => Ease.Functions.SineIn, 
			ShakeStrength.Medium => Ease.Functions.QuadIn, 
			ShakeStrength.Strong => Ease.Functions.QuartIn, 
			ShakeStrength.TooMuch => Ease.Functions.ExpoIn, 
			_ => throw new ArgumentOutOfRangeException("strength", strength, null), 
		};
	}

	private float SecondsForDuration(ShakeDuration duration)
	{
		return duration switch
		{
			ShakeDuration.Short => 0.15f, 
			ShakeDuration.Normal => 0.3f, 
			ShakeDuration.Long => 0.6f, 
			ShakeDuration.Forever => 2f, 
			_ => throw new ArgumentOutOfRangeException("duration", duration, null), 
		};
	}
}
