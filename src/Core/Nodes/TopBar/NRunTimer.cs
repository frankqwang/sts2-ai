using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Screens.Capstones;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.TopBar;

public partial class NRunTimer : Control
{
	private MegaLabel _timerLabel;

	private Timer _timer;

	public override void _Ready()
	{
		_timerLabel = GetNode<MegaLabel>("TimerLabel");
		ToggleTimer(on: false);
		CallDeferred("DeferredInit");
	}

	private void DeferredInit()
	{
		NMapScreen.Instance.Connect(CanvasItem.SignalName.VisibilityChanged, Callable.From(RefreshVisibility));
		NCapstoneContainer.Instance.Connect(NCapstoneContainer.SignalName.Changed, Callable.From(RefreshVisibility));
		_timer = new Timer();
		_timer.WaitTime = 1.0;
		_timer.Autostart = false;
		_timer.Connect(Timer.SignalName.Timeout, Callable.From(OnTimerTimeout));
		this.AddChildSafely(_timer);
		_timer.Start();
	}

	public override void _ExitTree()
	{
		_timer.Stop();
	}

	public void RefreshVisibility()
	{
		if (SaveManager.Instance.PrefsSave.ShowRunTimer)
		{
			ToggleTimer(on: true);
		}
		else
		{
			ToggleTimer(NCapstoneContainer.Instance.InUse || NMapScreen.Instance.Visible);
		}
	}

	private void ToggleTimer(bool on)
	{
		base.Visible = on;
	}

	private void OnTimerTimeout()
	{
		if (!RunManager.Instance.IsGameOver)
		{
			_timerLabel.SetTextAutoSize(TimeFormatting.Format(RunManager.Instance.RunTime));
		}
	}
}
