using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Potions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Timeline.UnlockScreens;

public partial class NUnlockPotionsScreen : NUnlockScreen
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("timeline_screen/unlock_potions_screen");

	private Control _potionRow;

	private NCommonBanner _banner;

	private IReadOnlyList<PotionModel> _potions;

	private Tween? _potionTween;

	private const float _potionXOffset = 350f;

	private static readonly Vector2 _potionScale = Vector2.One * 3f;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public static NUnlockPotionsScreen Create()
	{
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NUnlockPotionsScreen>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		ConnectSignals();
		_banner = GetNode<NCommonBanner>("%Banner");
		_banner.label.SetTextAutoSize(new LocString("timeline", "UNLOCK_POTIONS_BANNER").GetRawText());
		_banner.AnimateIn();
		_potionRow = GetNode<Control>("%PotionRow");
		LocString locString = new LocString("timeline", "UNLOCK_POTIONS");
		GetNode<MegaRichTextLabel>("%ExplanationText").Text = "[center]" + locString.GetFormattedText() + "[/center]";
	}

	public override void Open()
	{
		base.Open();
		SfxCmd.Play("event:/sfx/ui/timeline/ui_timeline_unlock");
		_potionTween = CreateTween().SetParallel();
		int num = -1;
		foreach (PotionModel potion in _potions)
		{
			NPotionHolder nPotionHolder = NPotionHolder.Create(isUsable: false);
			_potionRow.AddChildSafely(nPotionHolder);
			nPotionHolder.Position = base.Position;
			nPotionHolder.Modulate = StsColors.transparentBlack;
			nPotionHolder.Scale = _potionScale;
			NPotion nPotion = NPotion.Create(potion.ToMutable());
			nPotionHolder.AddPotion(nPotion);
			nPotion.Position = Vector2.Zero;
			_potionTween.TweenProperty(nPotionHolder, "position", base.Position + Vector2.Right * 350f * num, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			_potionTween.TweenProperty(nPotionHolder, "modulate", Colors.White, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
			num++;
		}
	}

	public void SetPotions(IReadOnlyList<PotionModel> potions)
	{
		_potions = potions;
	}

	protected override void OnScreenClose()
	{
		NTimelineScreen.Instance.EnableInput();
	}
}
