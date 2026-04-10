using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Map;

public partial class NMapEraseButton : NButton
{
	private static readonly StringName _imagePath = "res://images/packed/map/drawing_eraser.png";

	private static readonly StringName _glowImagePath = "res://images/packed/map/drawing_eraser_glow.png";

	private Control _drawingToolHolder;

	private TextureRect _icon;

	private HoverTip _hoverTip;

	private Tween? _tween;

	private bool _isErasing;

	private static readonly Color _activeColor = new Color("FF5757FF");

	private static readonly Color _inactiveColor = new Color("FFFFFF80");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlyArray<string>(new string[2] { _imagePath, _glowImagePath });

	public override void _Ready()
	{
		ConnectSignals();
		_drawingToolHolder = (Control)GetParent();
		_icon = GetNode<TextureRect>("Icon");
		NControllerManager.Instance.Connect(NControllerManager.SignalName.ControllerDetected, Callable.From(OnControllerUpdated));
		NControllerManager.Instance.Connect(NControllerManager.SignalName.MouseDetected, Callable.From(OnControllerUpdated));
		OnControllerUpdated();
	}

	public void SetIsErasing(bool isErasing)
	{
		_isErasing = isErasing;
		_icon.Texture = PreloadManager.Cache.GetTexture2D(isErasing ? _glowImagePath : _imagePath);
		_icon.SelfModulate = (isErasing ? _activeColor : _inactiveColor);
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_icon, "scale", Vector2.One * 1.2f, 0.05);
		_tween.TweenProperty(_icon, "self_modulate", _activeColor, 0.05);
		NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(_drawingToolHolder, _hoverTip);
		nHoverTipSet.GlobalPosition = _drawingToolHolder.GlobalPosition + new Vector2(10f, -132f);
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_icon, "scale", Vector2.One * 1.1f, 0.05);
		_tween.TweenProperty(_icon, "self_modulate", _isErasing ? _activeColor : _inactiveColor, 0.05);
		NHoverTipSet.Remove(_drawingToolHolder);
	}

	private void OnControllerUpdated()
	{
		LocString description = new LocString("map", "ERASING_BUTTON.description");
		LocString title = ((!NControllerManager.Instance.IsUsingController) ? new LocString("map", "ERASING_BUTTON.title_mkb") : new LocString("map", "ERASING_BUTTON.title_controller"));
		_hoverTip = new HoverTip(title, description);
	}
}
