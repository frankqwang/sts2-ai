using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;

public partial class NJoinFriendRefreshButton : NButton
{
	private static readonly StringName _v = new StringName("v");

	private static readonly Vector2 _hoverScale = Vector2.One * 1.01f;

	private static readonly Vector2 _pressScale = Vector2.One * 0.99f;

	private const float _defaultV = 0.9f;

	private const float _hoverV = 1.2f;

	private Tween? _tween;

	private ShaderMaterial _hsv;

	protected override string[] Hotkeys => new string[1] { MegaInput.accept };

	public ulong PlayerId { get; private set; }

	public override void _Ready()
	{
		ConnectSignals();
		MegaLabel node = GetNode<MegaLabel>("Label");
		node.SetTextAutoSize(new LocString("main_menu_ui", "JOIN_FRIENDS_MENU.refresh").GetFormattedText());
		_hsv = (ShaderMaterial)base.Material;
	}

	protected override void OnPress()
	{
		base.OnPress();
		base.Scale = _pressScale;
	}

	protected override void OnRelease()
	{
		base.OnRelease();
		base.Scale = Vector2.One;
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_tween?.Kill();
		base.Scale = _hoverScale;
		_hsv.SetShaderParameter(_v, 1.2f);
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderParam), 1.2f, 0.9f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	private void UpdateShaderParam(float newV)
	{
		_hsv.SetShaderParameter(_v, newV);
	}
}
