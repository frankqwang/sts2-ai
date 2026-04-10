using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Cards;

public partial class NTinyCard : NButton
{
	private TextureRect _cardBack;

	private TextureRect _cardPortrait;

	private TextureRect _cardPortraitShadow;

	private Control _cardBanner;

	public override void _Ready()
	{
		ConnectSignals();
		_cardBack = GetNode<TextureRect>("%CardBack");
		_cardPortrait = GetNode<TextureRect>("%Portrait");
		_cardPortraitShadow = GetNode<TextureRect>("%PortraitShadow");
		_cardBanner = GetNode<Control>("%Banner");
	}

	public void SetCard(CardModel card)
	{
		SetCardBackColor(card.Pool);
		SetCardPortraitShape(card.Type);
		SetBannerColor(card.Rarity);
		_cardBack.Material = card.FrameMaterial;
	}

	public void Set(CardPoolModel cardPool, CardType type, CardRarity rarity)
	{
		SetCardBackColor(cardPool);
		SetCardPortraitShape(type);
		SetBannerColor(rarity);
		_cardBack.Material = cardPool.AllCards.First().FrameMaterial;
	}

	private void SetCardBackColor(CardPoolModel cardPool)
	{
		_cardBack.Modulate = cardPool.DeckEntryCardColor;
	}

	private void SetCardPortraitShape(CardType type)
	{
		switch (type)
		{
		case CardType.Attack:
			_cardPortrait.Texture = PreloadManager.Cache.GetCompressedTexture2D("res://images/packed/run_history/attack_portrait.png");
			_cardPortraitShadow.Texture = PreloadManager.Cache.GetCompressedTexture2D("res://images/packed/run_history/attack_portrait_shadow.png");
			break;
		case CardType.Power:
			_cardPortrait.Texture = PreloadManager.Cache.GetCompressedTexture2D("res://images/packed/run_history/power_portrait.png");
			_cardPortraitShadow.Texture = PreloadManager.Cache.GetCompressedTexture2D("res://images/packed/run_history/power_portrait_shadow.png");
			break;
		default:
			_cardPortrait.Texture = PreloadManager.Cache.GetCompressedTexture2D("res://images/packed/run_history/skill_portrait.png");
			_cardPortraitShadow.Texture = PreloadManager.Cache.GetCompressedTexture2D("res://images/packed/run_history/skill_portrait_shadow.png");
			break;
		}
	}

	private void SetBannerColor(CardRarity rarity)
	{
		_cardBanner.Modulate = GetBannerColor(rarity);
	}

	private Color GetBannerColor(CardRarity rarity)
	{
		switch (rarity)
		{
		case CardRarity.Basic:
		case CardRarity.Common:
			return new Color("9C9C9CFF");
		case CardRarity.Uncommon:
			return new Color("64FFFFFF");
		case CardRarity.Rare:
			return new Color("FFDA36FF");
		case CardRarity.Curse:
			return new Color("E669FFFF");
		case CardRarity.Event:
			return new Color("13BE1AFF");
		case CardRarity.Quest:
			return new Color("F46836FF");
		default:
			Log.Warn($"Unspecified Rarity: {rarity}");
			return Colors.White;
		}
	}
}
