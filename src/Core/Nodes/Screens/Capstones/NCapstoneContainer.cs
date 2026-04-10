using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Capstones;

public partial class NCapstoneContainer : Control
{
	[Signal]
	public delegate void ChangedEventHandler();

	[Signal]
	public delegate void CapstoneClosedEventHandler();

	private Control _backstop;

	private Tween? _backstopFade;

	public ICapstoneScreen? CurrentCapstoneScreen { get; private set; }

	public bool InUse => CurrentCapstoneScreen != null;

	public static NCapstoneContainer? Instance => NRun.Instance?.GlobalUi.CapstoneContainer;

	public override void _Ready()
	{
		_backstop = GetNode<Control>("CapstoneBackstop");
		_backstop.Modulate = Colors.Transparent;
	}

	public override void _EnterTree()
	{
		ActiveScreenContext.Instance.Updated += OnActiveScreenChanged;
	}

	public override void _ExitTree()
	{
		ActiveScreenContext.Instance.Updated -= OnActiveScreenChanged;
	}

	public void Open(ICapstoneScreen screen)
	{
		NHoverTipSet.Clear();
		bool flag = CurrentCapstoneScreen != null;
		if (flag)
		{
			CloseInternal();
		}
		_backstopFade?.Kill();
		NOverlayStack.Instance.HideOverlays();
		if (!screen.UseSharedBackstop)
		{
			_backstop.Modulate = Colors.Transparent;
		}
		else if (flag || NOverlayStack.Instance.ScreenCount > 0)
		{
			_backstop.Modulate = Colors.White;
		}
		else
		{
			_backstopFade = CreateTween();
			_backstopFade.TweenProperty(_backstop, "modulate:a", 1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		}
		CurrentCapstoneScreen = screen;
		if (!GetChildren().Contains((Node)screen))
		{
			this.AddChildSafely((Node)screen);
		}
		((Node)screen).ProcessMode = ProcessModeEnum.Inherit;
		screen.AfterCapstoneOpened();
		if (RunManager.Instance.IsSinglePlayerOrFakeMultiplayer)
		{
			CombatManager.Instance.Pause();
		}
		ActiveScreenContext.Instance.Update();
		EmitSignal(SignalName.Changed);
	}

	public void Close()
	{
		if (CurrentCapstoneScreen != null)
		{
			CloseInternal();
			ActiveScreenContext.Instance.Update();
			EmitSignal(SignalName.CapstoneClosed);
			EmitSignal(SignalName.Changed);
		}
	}

	private void CloseInternal()
	{
		if (RunManager.Instance.IsSinglePlayerOrFakeMultiplayer)
		{
			CombatManager.Instance.Unpause();
		}
		NOverlayStack.Instance.ShowOverlays();
		if (NOverlayStack.Instance.ScreenCount > 0)
		{
			_backstop.Modulate = Colors.Transparent;
		}
		else
		{
			_backstopFade?.Kill();
			_backstopFade = CreateTween();
			_backstopFade.TweenProperty(_backstop, "modulate:a", 0f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		}
		ICapstoneScreen currentCapstoneScreen = CurrentCapstoneScreen;
		CurrentCapstoneScreen = null;
		if (currentCapstoneScreen is Node node)
		{
			node.ProcessMode = ProcessModeEnum.Disabled;
		}
		currentCapstoneScreen?.AfterCapstoneClosed();
		NHoverTipSet.Clear();
	}

	public void DisableBackstopInstantly()
	{
		_backstopFade?.Kill();
		_backstop.Modulate = Colors.Transparent;
	}

	public void EnableBackstopInstantly()
	{
		_backstopFade?.Kill();
		_backstop.Modulate = Colors.White;
	}

	public void CleanUp()
	{
		if (RunManager.Instance.IsSinglePlayerOrFakeMultiplayer)
		{
			CombatManager.Instance.Unpause();
		}
	}

	private void OnActiveScreenChanged()
	{
		if (InUse)
		{
			if (ActiveScreenContext.Instance.IsCurrent(CurrentCapstoneScreen))
			{
				base.FocusBehaviorRecursive = FocusBehaviorRecursiveEnum.Enabled;
			}
			else
			{
				base.FocusBehaviorRecursive = FocusBehaviorRecursiveEnum.Disabled;
			}
		}
	}
}
