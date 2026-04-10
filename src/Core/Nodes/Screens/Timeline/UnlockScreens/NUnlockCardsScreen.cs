using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.CommonUi;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Timeline.UnlockScreens;

public partial class NUnlockCardsScreen : NUnlockScreen
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("timeline_screen/unlock_cards_screen");

	private Control _cardRow;

	private NCommonBanner _banner;

	private readonly List<NCard> _nodes = new List<NCard>();

	private IReadOnlyList<CardModel> _cards;

	private Tween? _cardTween;

	private const float _cardXOffset = 350f;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public static NUnlockCardsScreen Create()
	{
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NUnlockCardsScreen>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		ConnectSignals();
		_banner = GetNode<NCommonBanner>("%Banner");
		_banner.label.SetTextAutoSize(new LocString("timeline", "UNLOCK_CARDS_BANNER").GetRawText());
		_banner.AnimateIn();
		_cardRow = GetNode<Control>("%CardRow");
		LocString locString = new LocString("timeline", "UNLOCK_CARDS");
		GetNode<MegaRichTextLabel>("%ExplanationText").Text = "[center]" + locString.GetFormattedText() + "[/center]";
	}

	public override void Open()
	{
		base.Open();
		SfxCmd.Play("event:/sfx/ui/timeline/ui_timeline_unlock");
		Vector2 vector = Vector2.Left * (_cards.Count - 1) * 350f * 0.5f;
		_cardTween = CreateTween().SetParallel();
		int num = 0;
		foreach (CardModel card in _cards)
		{
			NCard nCard = NCard.Create(card);
			NGridCardHolder nGridCardHolder = NGridCardHolder.Create(nCard);
			_cardRow.AddChildSafely(nGridCardHolder);
			nCard.UpdateVisuals(PileType.None, CardPreviewMode.Normal);
			nGridCardHolder.Scale = nGridCardHolder.SmallScale;
			_cardTween.TweenProperty(nGridCardHolder, "position", nGridCardHolder.Position + vector + Vector2.Right * 350f * num, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			_cardTween.TweenProperty(nGridCardHolder, "modulate", Colors.White, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic)
				.From(Colors.Black);
			nCard.ActivateRewardScreenGlow();
			_nodes.Add(nCard);
			num++;
		}
	}

	public void SetCards(IReadOnlyList<CardModel> cards)
	{
		_cards = cards;
	}

	protected override void OnScreenPreClose()
	{
		foreach (NCard node in _nodes)
		{
			node.KillRarityGlow();
		}
	}

	protected override void OnScreenClose()
	{
		NTimelineScreen.Instance.EnableInput();
	}

	public override void _ExitTree()
	{
		foreach (NGridCardHolder item in _cardRow.GetChildren().OfType<NGridCardHolder>())
		{
			item.QueueFreeSafely();
		}
	}
}
