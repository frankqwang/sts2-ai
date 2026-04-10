using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Map;

public partial class NMapClearButton : NButton
{
	private static readonly StringName _imagePath = "res://images/packed/map/drawing_clear.png";

	private static readonly StringName _glowImagePath = "res://images/packed/map/drawing_clear_glow.png";

	private Control _drawingToolHolder;

	private TextureRect _icon;

	private HoverTip _hoverTip;

	private Tween? _tween;

	private static readonly Color _activeColor = new Color("FFE57DFF");

	private static readonly Color _inactiveColor = new Color("FFFFFF80");

	public override void _Ready()
	{
		ConnectSignals();
		_drawingToolHolder = (Control)GetParent();
		_icon = GetNode<TextureRect>("Icon");
		_hoverTip = new HoverTip(new LocString("map", "CLEAR_DRAWING.title"), new LocString("map", "CLEAR_DRAWING.description"));
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_icon.Texture = PreloadManager.Cache.GetTexture2D(_glowImagePath);
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
		_icon.Texture = PreloadManager.Cache.GetTexture2D(_imagePath);
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_icon, "scale", Vector2.One * 1.1f, 0.05);
		_tween.TweenProperty(_icon, "self_modulate", _inactiveColor, 0.05);
		NHoverTipSet.Remove(_drawingToolHolder);
	}
}
