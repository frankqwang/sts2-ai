using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Cards.Holders;

public partial class NPreviewCardHolder : NCardHolder
{
	private bool _showHoverTips;

	private bool _scaleOnHover;

	private Vector2 _originalScale = Vector2.One;

	protected override Vector2 HoverScale => _originalScale * 1.1f;

	public override Vector2 SmallScale => _originalScale;

	public override bool IsShowingUpgradedCard
	{
		get
		{
			if (!base.IsShowingUpgradedCard)
			{
				if (CardModel != null)
				{
					return CardModel.UpgradePreviewType.IsPreview();
				}
				return false;
			}
			return true;
		}
	}

	private static string ScenePath => SceneHelper.GetScenePath("cards/holders/preview_card_holder");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	public static NPreviewCardHolder? Create(NCard card, bool showHoverTips, bool scaleOnHover)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NPreviewCardHolder nPreviewCardHolder = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NPreviewCardHolder>(PackedScene.GenEditState.Disabled);
		nPreviewCardHolder.Initialize(card, showHoverTips, scaleOnHover);
		return nPreviewCardHolder;
	}

	public override void _Ready()
	{
		ConnectSignals();
	}

	private void Initialize(NCard card, bool showHoverTips, bool scaleOnHover)
	{
		base.Name = $"UpgradePreviewCardHolder-{card.Model?.Id}";
		SetCard(card);
		base.Scale = SmallScale;
		_showHoverTips = showHoverTips;
		_scaleOnHover = scaleOnHover;
	}

	protected override void OnFocus()
	{
		_isHovered = true;
		if (_scaleOnHover)
		{
			_hoverTween?.Kill();
			base.Scale = HoverScale;
		}
		if (_showHoverTips)
		{
			CreateHoverTips();
		}
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		if (_scaleOnHover)
		{
			_hoverTween?.Kill();
			_hoverTween = CreateTween();
			_hoverTween.TweenProperty(this, "scale", SmallScale, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		}
		if (_showHoverTips)
		{
			ClearHoverTips();
		}
	}

	public void SetCardScale(Vector2 scale)
	{
		_originalScale = scale;
		base.Scale = _originalScale;
	}

	protected override void CreateHoverTips()
	{
		if (base.CardNode != null)
		{
			NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, base.CardNode.Model.HoverTips);
			nHoverTipSet.SetAlignmentForCardHolder(this);
		}
	}

	public override void _ExitTree()
	{
		if (base.CardNode != null)
		{
			if (IsAncestorOf(base.CardNode))
			{
				base.CardNode.QueueFreeSafely();
			}
			base.CardNode = null;
		}
	}
}
