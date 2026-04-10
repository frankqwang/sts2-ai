using System;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Timeline.UnlockScreens;

public abstract partial class NUnlockScreen : Control
{
	private NUnlockConfirmButton? _unlockConfirmButton;

	private Tween? _tween;

	public override void _Ready()
	{
		if (GetType() != typeof(NUnlockScreen))
		{
			Log.Error($"{GetType()}");
			throw new InvalidOperationException("Don't call base._Ready()! Call ConnectSignals() instead.");
		}
		ConnectSignals();
	}

	protected void ConnectSignals()
	{
		_unlockConfirmButton = GetNode<NUnlockConfirmButton>("ConfirmButton");
		_unlockConfirmButton.Connect(NClickableControl.SignalName.Released, Callable.From<NClickableControl>(delegate
		{
			TaskHelper.RunSafely(Close());
		}));
	}

	public virtual void Open()
	{
		NTimelineScreen.Instance.DisableInput();
		_unlockConfirmButton?.Disable();
		_tween?.FastForwardToCompletion();
		_tween = CreateTween();
		_tween.TweenProperty(this, "modulate:a", 1f, 0.5);
		_tween.Chain().TweenCallback(Callable.From(delegate
		{
			_unlockConfirmButton?.Enable();
		}));
	}

	protected async Task Close()
	{
		Log.Info($"Closing: {base.Name}");
		_tween?.FastForwardToCompletion();
		OnScreenPreClose();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "modulate", StsColors.transparentBlack, 1.0);
		await ToSignal(_tween, Tween.SignalName.Finished);
		OnScreenClose();
		if (!NTimelineScreen.Instance.IsScreenQueued())
		{
			await NTimelineScreen.Instance.HideBackstopAndShowUi(showBackButton: true);
		}
		else
		{
			NTimelineScreen.Instance.OpenQueuedScreen();
		}
		this.QueueFreeSafely();
	}

	protected virtual void OnScreenPreClose()
	{
	}

	protected virtual void OnScreenClose()
	{
	}
}
