using System;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Map;

public partial class NMapLegendItem : NButton
{
	private TextureRect _icon;

	private HoverTip _hoverTip;

	private Tween? _scaleDownTween;

	private static readonly Vector2 _hoverScale = Vector2.One * 1.25f;

	private const float _unhoverAnimDur = 0.5f;

	private MapPointType _pointType;

	public override void _Ready()
	{
		ConnectSignals();
		SetLocalizedFields(base.Name);
		SetMapPointType(base.Name);
		_icon = GetNode<TextureRect>("Icon");
	}

	private void SetMapPointType(string name)
	{
		_pointType = name switch
		{
			"UnknownLegendItem" => MapPointType.Unknown, 
			"MerchantLegendItem" => MapPointType.Shop, 
			"TreasureLegendItem" => MapPointType.Treasure, 
			"RestSiteLegendItem" => MapPointType.RestSite, 
			"EnemyLegendItem" => MapPointType.Monster, 
			"EliteLegendItem" => MapPointType.Elite, 
			_ => throw new ArgumentOutOfRangeException("Unknown Node " + name + " when setting MapLegend localization."), 
		};
	}

	private void SetLocalizedFields(string name)
	{
		string text = name switch
		{
			"UnknownLegendItem" => "LEGEND_UNKNOWN", 
			"MerchantLegendItem" => "LEGEND_MERCHANT", 
			"TreasureLegendItem" => "LEGEND_TREASURE", 
			"RestSiteLegendItem" => "LEGEND_REST", 
			"EnemyLegendItem" => "LEGEND_ENEMY", 
			"EliteLegendItem" => "LEGEND_ELITE", 
			_ => throw new ArgumentOutOfRangeException("Unknown Node " + name + " when setting MapLegend localization."), 
		};
		GetNode<MegaLabel>("MegaLabel").SetTextAutoSize(new LocString("map", text + ".title").GetFormattedText());
		_hoverTip = new HoverTip(new LocString("map", text + ".hoverTip.title"), new LocString("map", text + ".hoverTip.description"));
	}

	protected override void OnFocus()
	{
		_scaleDownTween?.Kill();
		_icon.Scale = _hoverScale;
		NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, _hoverTip);
		Control parent = GetParent<Control>();
		nHoverTipSet.GlobalPosition = parent.GlobalPosition + new Vector2(parent.Size.X - nHoverTipSet.Size.X, parent.Size.Y);
		NMapScreen.Instance.HighlightPointType(_pointType);
	}

	protected override void OnUnfocus()
	{
		_scaleDownTween = CreateTween().SetParallel();
		_scaleDownTween.TweenProperty(_icon, "scale", Vector2.One, 0.5).From(_hoverScale).SetEase(Tween.EaseType.Out)
			.SetTrans(Tween.TransitionType.Expo);
		NMapScreen.Instance.HighlightPointType(MapPointType.Unassigned);
		NHoverTipSet.Remove(this);
	}
}
