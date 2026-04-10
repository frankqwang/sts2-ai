using System;
using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.UI;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Relics;

namespace MegaCrit.Sts2.Core.Nodes.Screens.RelicCollection;

public partial class NRelicCollectionEntry : NButton
{
	public static readonly string scenePath = SceneHelper.GetScenePath("screens/relic_collection/relic_collection_entry");

	public static readonly string lockedIconPath = ImageHelper.GetImagePath("packed/common_ui/locked_model.png");

	public RelicModel relic;

	private Control _relicHolder;

	private Control _relicNode;

	private Tween? _hoverTween;

	private static LocString UnknownHoverTipTitle => new LocString("main_menu_ui", "COMPENDIUM_RELIC_COLLECTION.unknown.title");

	private static LocString UnknownHoverTipDescription => new LocString("main_menu_ui", "COMPENDIUM_RELIC_COLLECTION.unknown.description");

	private static HoverTip UnknownHoverTip => new HoverTip(UnknownHoverTipTitle, UnknownHoverTipDescription);

	private static LocString LockedHoverTipTitle => new LocString("main_menu_ui", "COMPENDIUM_RELIC_COLLECTION.locked.title");

	private static LocString LockedHoverTipDescription => new LocString("main_menu_ui", "COMPENDIUM_RELIC_COLLECTION.locked.description");

	private static HoverTip LockedHoverTip => new HoverTip(LockedHoverTipTitle, LockedHoverTipDescription);

	public ModelVisibility ModelVisibility { get; set; }

	public static NRelicCollectionEntry Create(RelicModel relic, ModelVisibility visibility)
	{
		NRelicCollectionEntry nRelicCollectionEntry = PreloadManager.Cache.GetScene(scenePath).Instantiate<NRelicCollectionEntry>(PackedScene.GenEditState.Disabled);
		nRelicCollectionEntry.relic = relic;
		nRelicCollectionEntry.ModelVisibility = visibility;
		return nRelicCollectionEntry;
	}

	public override void _Ready()
	{
		ConnectSignals();
		_relicHolder = GetNode<Control>("RelicHolder");
		if (ModelVisibility == ModelVisibility.Locked)
		{
			TextureRect textureRect = new TextureRect();
			textureRect.ExpandMode = TextureRect.ExpandModeEnum.IgnoreSize;
			textureRect.StretchMode = TextureRect.StretchModeEnum.KeepAspect;
			textureRect.Texture = PreloadManager.Cache.GetTexture2D(lockedIconPath);
			textureRect.Size = Vector2.One * 68f;
			textureRect.PivotOffset = textureRect.Size * 0.5f;
			textureRect.Modulate = StsColors.gray;
			_relicHolder.AddChildSafely(textureRect);
			_relicNode = textureRect;
		}
		else
		{
			NRelic nRelic = NRelic.Create(relic.ToMutable(), NRelic.IconSize.Small);
			_relicHolder.AddChildSafely(nRelic);
			if (ModelVisibility == ModelVisibility.NotSeen)
			{
				nRelic.Icon.SelfModulate = StsColors.ninetyPercentBlack;
				nRelic.Outline.SelfModulate = StsColors.halfTransparentWhite;
			}
			else
			{
				foreach (RelicPoolModel allCharacterRelicPool in ModelDb.AllCharacterRelicPools)
				{
					if (allCharacterRelicPool.AllRelicIds.Contains(relic.Id))
					{
						TextureRect outline = nRelic.Outline;
						Color labOutlineColor = allCharacterRelicPool.LabOutlineColor;
						labOutlineColor.A = 0.66f;
						outline.SelfModulate = labOutlineColor;
						break;
					}
				}
			}
			_relicNode = nRelic;
		}
		_relicNode.MouseFilter = MouseFilterEnum.Ignore;
		_relicNode.FocusMode = FocusModeEnum.None;
	}

	protected override void OnRelease()
	{
		_hoverTween?.Kill();
		_hoverTween = CreateTween();
		_hoverTween.TweenProperty(_relicNode, "scale", Vector2.One, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		NHoverTipSet.Remove(this);
	}

	protected override void OnFocus()
	{
		_hoverTween?.Kill();
		_relicNode.Scale = Vector2.One * 1.25f;
		ModelVisibility modelVisibility = ModelVisibility;
		IEnumerable<IHoverTip> enumerable = default(IEnumerable<IHoverTip>);
		switch (modelVisibility)
		{
		case ModelVisibility.None:
			throw new ArgumentOutOfRangeException();
		case ModelVisibility.Visible:
			enumerable = relic.HoverTips;
			break;
		case ModelVisibility.NotSeen:
			enumerable = new global::_003C_003Ez__ReadOnlySingleElementList<IHoverTip>(UnknownHoverTip);
			break;
		case ModelVisibility.Locked:
			enumerable = new global::_003C_003Ez__ReadOnlySingleElementList<IHoverTip>(LockedHoverTip);
			break;
		default:
			throw new System.Runtime.CompilerServices.SwitchExpressionException(modelVisibility);
			break;
		}
		IEnumerable<IHoverTip> hoverTips = enumerable;
		NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, hoverTips, HoverTip.GetHoverTipAlignment(this));
		nHoverTipSet.SetFollowOwner();
	}

	protected override void OnUnfocus()
	{
		_hoverTween?.Kill();
		_hoverTween = CreateTween();
		_hoverTween.TweenProperty(_relicNode, "scale", Vector2.One, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		NHoverTipSet.Remove(this);
	}

	protected override void OnPress()
	{
		base.OnPress();
		_hoverTween?.Kill();
		_hoverTween = CreateTween();
		_hoverTween.TweenProperty(_relicNode, "scale", Vector2.One, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}
}
