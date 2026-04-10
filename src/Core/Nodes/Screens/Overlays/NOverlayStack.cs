using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Overlays;

public partial class NOverlayStack : Control
{
	[Signal]
	public delegate void ChangedEventHandler();

	private readonly List<IOverlayScreen> _overlays = new List<IOverlayScreen>();

	private Control _backstop;

	private Tween? _backstopFade;

	public static NOverlayStack? Instance => NRun.Instance?.GlobalUi.Overlays;

	public int ScreenCount => _overlays.Count;

	public override void _Ready()
	{
		_backstop = GetNode<Control>("OverlayBackstop");
		_backstop.Modulate = Colors.Transparent;
		_backstop.MouseFilter = MouseFilterEnum.Ignore;
		Callable.From(() => NMapScreen.Instance.Connect(NMapScreen.SignalName.Opened, Callable.From(HideOverlays))).CallDeferred();
		Callable.From(() => NMapScreen.Instance.Connect(NMapScreen.SignalName.Closed, Callable.From(ShowOverlays))).CallDeferred();
	}

	public override void _EnterTree()
	{
		ActiveScreenContext.Instance.Updated += OnActiveScreenChanged;
	}

	public override void _ExitTree()
	{
		ActiveScreenContext.Instance.Updated -= OnActiveScreenChanged;
		Clear();
	}

	public void Push(IOverlayScreen screen)
	{
		Peek()?.AfterOverlayHidden();
		this.AddChildSafely((Node)screen);
		_overlays.Add(screen);
		screen.AfterOverlayOpened();
		screen.AfterOverlayShown();
		_backstop.MouseFilter = MouseFilterEnum.Stop;
		_backstopFade?.Kill();
		MoveChild(_backstop, _overlays.IndexOf(screen));
		if (!screen.UseSharedBackstop)
		{
			_backstop.Modulate = Colors.Transparent;
		}
		else if (ScreenCount == 1)
		{
			ShowBackstop();
		}
		else
		{
			_backstop.Modulate = Colors.White;
		}
		ActiveScreenContext.Instance.Update();
		EmitSignal(SignalName.Changed);
	}

	public void Remove(IOverlayScreen screen)
	{
		bool flag = screen == Peek();
		if (flag)
		{
			HideBackstop();
			screen.AfterOverlayHidden();
		}
		screen.AfterOverlayClosed();
		_overlays.Remove(screen);
		if (flag)
		{
			IOverlayScreen overlayScreen = Peek();
			if (overlayScreen != null)
			{
				_backstop.MouseFilter = MouseFilterEnum.Stop;
				MoveChild(_backstop, _overlays.IndexOf(overlayScreen));
				if (overlayScreen.UseSharedBackstop)
				{
					_backstop.Modulate = Colors.White;
				}
				else
				{
					HideBackstop();
				}
				overlayScreen.AfterOverlayShown();
			}
			else
			{
				HideBackstop();
			}
		}
		ActiveScreenContext.Instance.Update();
		EmitSignal(SignalName.Changed);
	}

	public void Clear()
	{
		for (IOverlayScreen overlayScreen = Peek(); overlayScreen != null; overlayScreen = Peek())
		{
			Remove(overlayScreen);
		}
	}

	public void HideOverlays()
	{
		_backstop.Modulate = Colors.Transparent;
		Peek()?.AfterOverlayHidden();
	}

	public void ShowOverlays()
	{
		IOverlayScreen overlayScreen = Peek();
		if (overlayScreen != null && !NMapScreen.Instance.IsOpen)
		{
			_backstop.Modulate = (overlayScreen.UseSharedBackstop ? Colors.White : Colors.Transparent);
			overlayScreen.AfterOverlayShown();
		}
	}

	public void ShowBackstop()
	{
		IOverlayScreen? overlayScreen = Peek();
		if (overlayScreen == null || overlayScreen.UseSharedBackstop)
		{
			_backstop.MouseFilter = MouseFilterEnum.Stop;
			_backstopFade?.Kill();
			_backstopFade = CreateTween();
			_backstopFade.TweenProperty(_backstop, "modulate:a", 1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		}
	}

	public void HideBackstop()
	{
		IOverlayScreen? overlayScreen = Peek();
		if (overlayScreen == null || overlayScreen.UseSharedBackstop)
		{
			_backstop.MouseFilter = MouseFilterEnum.Ignore;
			_backstopFade?.Kill();
			if (ScreenCount <= 1)
			{
				_backstopFade = CreateTween();
				_backstopFade.TweenProperty(_backstop, "modulate:a", 0f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
			}
			else
			{
				_backstop.Modulate = Colors.Transparent;
			}
		}
	}

	public IOverlayScreen? Peek()
	{
		return _overlays.LastOrDefault();
	}

	private void OnActiveScreenChanged()
	{
		IOverlayScreen overlayScreen = Peek();
		if (overlayScreen != null)
		{
			if (ActiveScreenContext.Instance.IsCurrent(overlayScreen))
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
