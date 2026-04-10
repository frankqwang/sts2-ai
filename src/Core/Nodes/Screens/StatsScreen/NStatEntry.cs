using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;

namespace MegaCrit.Sts2.Core.Nodes.Screens.StatsScreen;

public partial class NStatEntry : NClickableControl
{
	private HoverTip? _hoverTip;

	private Tween? _tween;

	private string? _imgUrl;

	private TextureRect _icon;

	private MegaRichTextLabel _topLabel;

	private MegaRichTextLabel _bottomLabel;

	private NSelectionReticle _controllerFocusReticle;

	private static string ScenePath => SceneHelper.GetScenePath("screens/stats_screen/stats_screen_section");

	public static NStatEntry Create(string imgUrl)
	{
		NStatEntry nStatEntry = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NStatEntry>(PackedScene.GenEditState.Disabled);
		nStatEntry._imgUrl = imgUrl;
		return nStatEntry;
	}

	public void SetTopText(string text)
	{
		_topLabel.Visible = true;
		_topLabel.SetTextAutoSize(text);
	}

	public void SetBottomText(string text)
	{
		_bottomLabel.Visible = true;
		_bottomLabel.SetTextAutoSize(text);
	}

	public override void _Ready()
	{
		SetPivotOffset(new Vector2(50f, base.Size.Y * 0.5f));
		_icon = GetNode<TextureRect>("%Icon");
		if (_imgUrl != null)
		{
			_icon.Texture = PreloadManager.Cache.GetTexture2D(_imgUrl);
		}
		_topLabel = GetNode<MegaRichTextLabel>("%TopLabel");
		_bottomLabel = GetNode<MegaRichTextLabel>("%BottomLabel");
		_controllerFocusReticle = GetNode<NSelectionReticle>("%SelectionReticle");
		ConnectSignals();
	}

	public void SetHoverTip(HoverTip hoverTip)
	{
		_hoverTip = hoverTip;
	}

	protected override void OnFocus()
	{
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "scale", Vector2.One * 1.05f, 0.05);
		if (_hoverTip.HasValue)
		{
			if (base.GlobalPosition.X < GetViewport().GetVisibleRect().Size.X * 0.4f)
			{
				NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, _hoverTip);
				nHoverTipSet.GlobalPosition = new Vector2(base.GlobalPosition.X - 392f, base.GlobalPosition.Y);
			}
			else
			{
				NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, _hoverTip);
				nHoverTipSet.GlobalPosition = new Vector2(base.GlobalPosition.X + 532f, base.GlobalPosition.Y);
			}
		}
		if (NControllerManager.Instance.IsUsingController)
		{
			_controllerFocusReticle.OnSelect();
		}
	}

	protected override void OnUnfocus()
	{
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		if (_hoverTip.HasValue)
		{
			NHoverTipSet.Remove(this);
		}
		_controllerFocusReticle.OnDeselect();
	}
}
