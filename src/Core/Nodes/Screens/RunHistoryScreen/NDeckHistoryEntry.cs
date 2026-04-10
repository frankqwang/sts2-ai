using System;
using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.RunHistoryScreen;

public partial class NDeckHistoryEntry : NButton
{
	[Signal]
	public delegate void ClickedEventHandler(NDeckHistoryEntry entry);

	private MegaLabel _titleLabel;

	private NTinyCard _cardImage;

	private TextureRect _enchantmentImage;

	private MarginContainer _labelContainer;

	private Tween? _scaleTween;

	private int _amount;

	private static string ScenePath => SceneHelper.GetScenePath("screens/run_history_screen/deck_history_entry");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	public IEnumerable<int> FloorsAddedToDeck { get; private set; }

	public CardModel Card { get; private set; }

	public override void _Ready()
	{
		ConnectSignals();
		_titleLabel = GetNode<MegaLabel>("%Label");
		_labelContainer = _titleLabel.GetParent<MarginContainer>();
		_cardImage = GetNode<NTinyCard>("%Card");
		_enchantmentImage = GetNode<TextureRect>("%Enchantment");
		_cardImage.PivotOffset = _cardImage.Size * 0.5f;
		Reload();
	}

	public static NDeckHistoryEntry Create(CardModel card, int amount)
	{
		return Create(card, amount, Array.Empty<int>());
	}

	public static NDeckHistoryEntry Create(CardModel card, int amount, IEnumerable<int> floorsAdded)
	{
		NDeckHistoryEntry nDeckHistoryEntry = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NDeckHistoryEntry>(PackedScene.GenEditState.Disabled);
		nDeckHistoryEntry.Card = card;
		nDeckHistoryEntry._amount = amount;
		nDeckHistoryEntry.FloorsAddedToDeck = floorsAdded;
		return nDeckHistoryEntry;
	}

	private void Reload()
	{
		_titleLabel.SetTextAutoSize(Card.Title);
		bool flag = Card.CurrentUpgradeLevel >= 1;
		bool flag2 = Card.Enchantment != null;
		string text = Card.Title;
		if (_amount > 1)
		{
			text = $"{_amount}x {text}";
		}
		_titleLabel.SetTextAutoSize(text);
		if (flag2)
		{
			_titleLabel.AddThemeColorOverride(ThemeConstants.Label.fontColor, StsColors.purple);
		}
		else if (flag)
		{
			_titleLabel.AddThemeColorOverride(ThemeConstants.Label.fontColor, StsColors.green);
		}
		_cardImage.SetCard(Card);
		if (Card.Enchantment != null)
		{
			_enchantmentImage.Texture = Card.Enchantment.Icon;
		}
		_enchantmentImage.Visible = Card.Enchantment != null;
		base.Size = new Vector2(_cardImage.Size.X + _titleLabel.Size.X + 10f, base.Size.Y);
	}

	protected override void OnFocus()
	{
		_scaleTween?.FastForwardToCompletion();
		_scaleTween = CreateTween().SetParallel();
		_scaleTween.TweenProperty(_cardImage, "scale", Vector2.One * 1.5f, 0.05);
		_scaleTween.TweenProperty(_labelContainer, "position:x", _labelContainer.Position.X + 8f, 0.05);
	}

	protected override void OnUnfocus()
	{
		_scaleTween?.FastForwardToCompletion();
		_scaleTween = CreateTween().SetParallel();
		_scaleTween.TweenProperty(_cardImage, "scale", Vector2.One, 0.5).SetTrans(Tween.TransitionType.Expo).SetEase(Tween.EaseType.Out);
		_scaleTween.TweenProperty(_labelContainer, "position:x", _labelContainer.Position.X - 8f, 0.5).SetTrans(Tween.TransitionType.Expo).SetEase(Tween.EaseType.Out);
	}

	protected override void OnRelease()
	{
		EmitSignal(SignalName.Clicked, this);
	}
}
