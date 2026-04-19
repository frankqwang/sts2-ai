using System.Threading;
using System.Threading.Tasks;
using Godot;

namespace MegaCrit.Sts2.Core.Nodes.GodotExtensions;

public static class TweenHelper
{
	public static void FastForwardToCompletion(this Tween t)
	{
		t.CustomStep(999999999.0);
	}

	public static Task AwaitFinished(this Tween tween, Node owner)
	{
		if (!tween.IsValid() || !tween.IsRunning())
		{
			return Task.CompletedTask;
		}
		TaskCompletionSource tcs = new TaskCompletionSource();
		bool resolved = false;
		tween.Finished += OnFinished;
		owner.TreeExiting += OnExiting;
		return tcs.Task;
		void OnExiting()
		{
			if (!resolved)
			{
				resolved = true;
				tween.Finished -= OnFinished;
				tcs.TrySetCanceled();
			}
		}
		void OnFinished()
		{
			if (!resolved)
			{
				resolved = true;
				tween.Finished -= OnFinished;
				if (GodotObject.IsInstanceValid(owner))
				{
					owner.TreeExiting -= OnExiting;
				}
				tcs.TrySetResult();
			}
		}
	}

	public static Task AwaitFinished(this Tween tween, CancellationToken ct)
	{
		TaskCompletionSource tcs = new TaskCompletionSource();
		int unsubscribed = 0;
		CancellationTokenRegistration ctr = default(CancellationTokenRegistration);
		tween.Finished += OnFinished;
		if (ct.CanBeCanceled)
		{
			ctr = ct.Register(delegate
			{
				if (Interlocked.Exchange(ref unsubscribed, 1) == 0)
				{
					tween.Finished -= OnFinished;
				}
				tcs.TrySetCanceled(ct);
			});
		}
		return tcs.Task;
		void OnFinished()
		{
			if (Interlocked.Exchange(ref unsubscribed, 1) == 0)
			{
				tween.Finished -= OnFinished;
			}
			ctr.Dispose();
			tcs.TrySetResult();
		}
	}
}
