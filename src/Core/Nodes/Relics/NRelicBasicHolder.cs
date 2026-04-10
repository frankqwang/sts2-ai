using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Relics;

public partial class NRelicBasicHolder : NButton
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("relics/relic_basic_holder");

	private NRelic _relic;

	private Tween? _hoverTween;

	private RelicModel _model;

	public NRelic Relic => _relic;

	public static NRelicBasicHolder? Create(RelicModel relic)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NRelicBasicHolder nRelicBasicHolder = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NRelicBasicHolder>(PackedScene.GenEditState.Disabled);
		nRelicBasicHolder.Name = $"NRelicBasicHolder-{relic.Id}";
		nRelicBasicHolder._model = relic;
		return nRelicBasicHolder;
	}

	public override void _Ready()
	{
		ConnectSignals();
		_relic = GetNode<NRelic>("%Relic");
		_relic.Model = _model;
	}

	public override void _ExitTree()
	{
		_hoverTween?.Kill();
	}

	protected override void OnFocus()
	{
		_hoverTween?.Kill();
		_hoverTween = CreateTween();
		_hoverTween.TweenProperty(_relic.Icon, "scale", Vector2.One * 1.25f, 0.05);
		NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, _relic.Model.HoverTips);
		nHoverTipSet.SetAlignmentForRelic(_relic);
	}

	protected override void OnUnfocus()
	{
		_hoverTween?.Kill();
		_hoverTween = CreateTween();
		_hoverTween.TweenProperty(_relic.Icon, "scale", Vector2.One, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		NHoverTipSet.Remove(this);
	}
}
