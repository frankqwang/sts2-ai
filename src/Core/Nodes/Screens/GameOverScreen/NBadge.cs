using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models.Badges;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;

public partial class NBadge : NButton
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/game_over_screen/badge");

	private Tween? _tween;

	private HoverTip _hoverTip;

	private LocString _title;

	private LocString _description;

	private const string _table = "badges";

	private Control _hoverNode;

	public static NBadge? Create(Badge badgeModel)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NBadge nBadge = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NBadge>(PackedScene.GenEditState.Disabled);
		nBadge.GetNode<TextureRect>("%BadgeHolder").Texture = badgeModel.BadgeBase;
		nBadge.GetNode<TextureRect>("%Icon").Texture = badgeModel.BadgeIcon;
		if (LocString.Exists("badges", badgeModel.Id + "." + GetRarityPrefix(badgeModel.Rarity) + "Title"))
		{
			nBadge._title = new LocString("badges", badgeModel.Id + "." + GetRarityPrefix(badgeModel.Rarity) + "Title");
			nBadge._description = new LocString("badges", badgeModel.Id + "." + GetRarityPrefix(badgeModel.Rarity) + "Description");
		}
		else
		{
			nBadge._title = new LocString("badges", badgeModel.Id + ".title");
			nBadge._description = new LocString("badges", badgeModel.Id + ".description");
		}
		return nBadge;
	}

	public static NBadge? Create(string id, BadgeRarity rarity)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NBadge nBadge = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NBadge>(PackedScene.GenEditState.Disabled);
		nBadge.GetNode<TextureRect>("%BadgeHolder").Texture = GetBadgeBaseTexture(rarity);
		nBadge.GetNode<TextureRect>("%Icon").Texture = PreloadManager.Cache.GetTexture2D(ImageHelper.GetImagePath("ui/game_over_screen/badge_" + id.ToLowerInvariant() + ".png"));
		if (LocString.Exists("badges", id + "." + GetRarityPrefix(rarity) + "Title"))
		{
			nBadge._title = new LocString("badges", id + "." + GetRarityPrefix(rarity) + "Title");
			nBadge._description = new LocString("badges", id + "." + GetRarityPrefix(rarity) + "Description");
		}
		else
		{
			nBadge._title = new LocString("badges", id + ".title");
			nBadge._description = new LocString("badges", id + ".description");
		}
		return nBadge;
	}

	public override void _Ready()
	{
		ConnectSignals();
		_hoverNode = this;
		_hoverTip = new HoverTip(_title, _description);
	}

	private static string GetRarityPrefix(BadgeRarity rarity)
	{
		return rarity switch
		{
			BadgeRarity.Bronze => "bronze", 
			BadgeRarity.Silver => "silver", 
			BadgeRarity.Gold => "gold", 
			_ => "ERROR", 
		};
	}

	public async Task AnimateIn()
	{
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "modulate:a", 1f, 0.25);
		_tween.TweenProperty(this, "position:y", base.Position.Y, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Spring)
			.From(base.Position.Y + 40f);
		await ToSignal(_tween, Tween.SignalName.Finished);
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(_hoverNode, _hoverTip);
		nHoverTipSet.GlobalPosition = _hoverNode.GlobalPosition + new Vector2(10f, -132f);
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		NHoverTipSet.Remove(_hoverNode);
	}

	private static Texture2D GetBadgeBaseTexture(BadgeRarity rarity)
	{
		return rarity switch
		{
			BadgeRarity.Bronze => PreloadManager.Cache.GetTexture2D(ImageHelper.GetImagePath("ui/game_over_screen/badge_bronze.png")), 
			BadgeRarity.Silver => PreloadManager.Cache.GetTexture2D(ImageHelper.GetImagePath("ui/game_over_screen/badge_silver.png")), 
			BadgeRarity.Gold => PreloadManager.Cache.GetTexture2D(ImageHelper.GetImagePath("ui/game_over_screen/badge_gold.png")), 
			_ => PreloadManager.Cache.GetTexture2D(ImageHelper.GetImagePath("atlases/power_atlas.sprites/missing_power.tres")), 
		};
	}
}
