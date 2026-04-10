using Godot;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Screens.Capstones;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;

namespace MegaCrit.Sts2.Core.Nodes.TopBar;

public partial class NTopBarMapButton : NTopBarButton
{
	private static readonly StringName _v = new StringName("v");

	private static readonly HoverTip _hoverTip = new HoverTip(new LocString("static_hover_tips", "MAP.title"), new LocString("static_hover_tips", "MAP.description"));

	private const float _defaultV = 0.9f;

	private Tween? _oscillateTween;

	protected override string[] Hotkeys => new string[1] { MegaInput.viewMap };

	protected override void OnRelease()
	{
		base.OnRelease();
		if (IsOpen())
		{
			NCapstoneContainer? instance = NCapstoneContainer.Instance;
			if (instance != null && instance.InUse)
			{
				NCapstoneContainer.Instance.Close();
			}
			else
			{
				NMapScreen.Instance.Close();
			}
		}
		else
		{
			NCapstoneContainer.Instance.Close();
			NMapScreen.Instance.Open(isOpenedFromTopBar: true);
		}
		_hsv?.SetShaderParameter(_v, 0.9f);
	}

	protected override bool IsOpen()
	{
		return NMapScreen.Instance.Visible;
	}

	public void StartOscillation()
	{
		_oscillateTween?.Kill();
		_oscillateTween = CreateTween();
		_oscillateTween.SetLoops();
		_oscillateTween.TweenProperty(_icon, "rotation", -0.12f, 0.8).SetTrans(Tween.TransitionType.Sine).SetEase(Tween.EaseType.InOut);
		_oscillateTween.TweenProperty(_icon, "rotation", 0.12f, 0.8).SetTrans(Tween.TransitionType.Sine).SetEase(Tween.EaseType.InOut);
	}

	public void StopOscillation()
	{
		_oscillateTween?.Kill();
		_oscillateTween = CreateTween();
		_oscillateTween.TweenProperty(_icon, "rotation", 0f, 0.5).SetTrans(Tween.TransitionType.Spring).SetEase(Tween.EaseType.Out);
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, _hoverTip);
		nHoverTipSet.GlobalPosition = base.GlobalPosition + new Vector2(base.Size.X - nHoverTipSet.Size.X, base.Size.Y + 20f);
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		NHoverTipSet.Remove(this);
	}
}
