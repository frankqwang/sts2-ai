using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Enchantments;
using MegaCrit.Sts2.Core.Entities.UI;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Helpers.Models;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Pooling;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Vfx;
using MegaCrit.Sts2.Core.Nodes.Vfx.Cards;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Cards;

public partial class NCard : Control, IPoolable
{
	private const string _scenePath = "res://scenes/cards/card.tscn";

	private static readonly string _portraitBlurMaterialPath = "res://scenes/cards/card_portrait_blur_material.tres";

	private static readonly string _canvasGroupMaskMaterialPath = "res://scenes/cards/card_canvas_group_mask_material.tres";

	private static readonly string _canvasGroupBlurMaterialPath = "res://scenes/cards/card_canvas_group_blur_material.tres";

	private static readonly string _canvasGroupMaskBlurMaterialPath = "res://scenes/cards/card_canvas_group_mask_blur_material.tres";

	private static readonly float _typePlaqueXMargin = 17f;

	private static readonly float _typePlaqueMinXSize = 61f;

	private static readonly StringName _v = new StringName("v");

	private static readonly StringName _s = new StringName("s");

	private static readonly StringName _h = new StringName("h");

	public static readonly Vector2 defaultSize = new Vector2(300f, 422f);

	private CardModel? _model;

	private MegaLabel _titleLabel;

	private MegaRichTextLabel _descriptionLabel;

	private TextureRect _ancientPortrait;

	private TextureRect _portrait;

	private TextureRect _frame;

	private TextureRect _ancientBorder;

	private Control _ancientBanner;

	private TextureRect _ancientTextBg;

	private TextureRect _ancientHighlight;

	private TextureRect _portraitBorder;

	private TextureRect _banner;

	private TextureRect _lock;

	private NinePatchRect _typePlaque;

	private MegaLabel _typeLabel;

	private CanvasGroup _portraitCanvasGroup;

	private NCardRareGlow? _rareGlow;

	private NCardUncommonGlow? _uncommonGlow;

	private GpuParticles2D _sparkles;

	private readonly List<NRelicFlashVfx> _flashVfx = new List<NRelicFlashVfx>();

	private TextureRect _energyIcon;

	private MegaLabel _energyLabel;

	private TextureRect _unplayableEnergyIcon;

	private TextureRect _starIcon;

	private MegaLabel _starLabel;

	private TextureRect _unplayableStarIcon;

	private Node _overlayContainer;

	private Control? _cardOverlay;

	private Creature? _previewTarget;

	private Control _enchantmentTab;

	private TextureRect _enchantmentVfxOverride;

	private TextureRect _enchantmentIcon;

	private MegaLabel _enchantmentLabel;

	private Vector2 _defaultEnchantmentPosition;

	private const int _enchantmentTabStarLabelOffset = 45;

	private EnchantmentModel? _subscribedEnchantment;

	private bool _pretendCardCanBePlayed;

	private bool _forceUnpoweredPreview;

	private Material? _portraitBlurMaterial;

	private Material? _canvasGroupMaskBlurMaterial;

	private Material? _canvasGroupBlurMaterial;

	private Material? _canvasGroupMaskMaterial;

	private ModelVisibility _visibility = ModelVisibility.Visible;

	private readonly LocString _unknownDescription = new LocString("card_library", "UNKNOWN.description");

	private readonly LocString _unknownTitle = new LocString("card_library", "UNKNOWN.title");

	private readonly LocString _lockedDescription = new LocString("card_library", "LOCKED.description");

	private readonly LocString _lockedTitle = new LocString("card_library", "LOCKED.title");

	public NCardHighlight CardHighlight { get; private set; }

	public Control Body { get; private set; }

	public ModelVisibility Visibility
	{
		get
		{
			return _visibility;
		}
		set
		{
			if (_visibility != value)
			{
				_visibility = value;
				Reload();
			}
		}
	}

	public Tween? PlayPileTween { get; set; }

	private Tween? RandomizeCostTween { get; set; }

	public PileType DisplayingPile { get; private set; }

	public Control EnchantmentTab => _enchantmentTab;

	public TextureRect EnchantmentVfxOverride => _enchantmentVfxOverride;

	public CardModel? Model
	{
		get
		{
			return _model;
		}
		set
		{
			if (_model != value)
			{
				CardModel model = _model;
				UnsubscribeFromModel(model);
				_model = value;
				Reload();
				SubscribeToModel(_model);
				this.ModelChanged?.Invoke(model);
				if (_model != null && (_model.RunState != null || _model.CombatState != null) && LocalContext.IsMine(_model))
				{
					SaveManager.Instance.MarkCardAsSeen(_model);
				}
			}
		}
	}

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlyArray<string>(new string[5] { "res://scenes/cards/card.tscn", _portraitBlurMaterialPath, _canvasGroupBlurMaterialPath, _canvasGroupMaskBlurMaterialPath, _canvasGroupMaskMaterialPath });

	public event Action<CardModel?>? ModelChanged;

	public void OnInstantiated()
	{
	}

	public override void _Ready()
	{
		_titleLabel = GetNode<MegaLabel>("%TitleLabel");
		_descriptionLabel = GetNode<MegaRichTextLabel>("%DescriptionLabel");
		_frame = GetNode<TextureRect>("%Frame");
		_ancientBorder = GetNode<TextureRect>("%AncientBorder");
		_ancientTextBg = GetNode<TextureRect>("%AncientTextBg");
		_ancientBanner = GetNode<Control>("%AncientBanner");
		_ancientHighlight = GetNode<TextureRect>("%AncientHighlight");
		_portrait = GetNode<TextureRect>("%Portrait");
		_ancientPortrait = GetNode<TextureRect>("%AncientPortrait");
		_typeLabel = GetNode<MegaLabel>("%TypeLabel");
		_portraitBorder = GetNode<TextureRect>("%PortraitBorder");
		_portraitCanvasGroup = GetNode<CanvasGroup>("%PortraitCanvasGroup");
		_energyLabel = GetNode<MegaLabel>("%EnergyLabel");
		_energyIcon = GetNode<TextureRect>("%EnergyIcon");
		_starLabel = GetNode<MegaLabel>("%StarLabel");
		_starIcon = GetNode<TextureRect>("%StarIcon");
		_banner = GetNode<TextureRect>("%TitleBanner");
		_lock = GetNode<TextureRect>("%Lock");
		_typePlaque = GetNode<NinePatchRect>("%TypePlaque");
		_unplayableEnergyIcon = GetNode<TextureRect>("%UnplayableEnergyIcon");
		_unplayableStarIcon = GetNode<TextureRect>("%UnplayableStarIcon");
		_enchantmentIcon = GetNode<TextureRect>("%Enchantment/Icon");
		_enchantmentLabel = GetNode<MegaLabel>("%Enchantment/Label");
		_sparkles = GetNode<GpuParticles2D>("CardContainer/CardSparkles");
		_enchantmentTab = GetNode<Control>("%Enchantment");
		_enchantmentTab.Visible = false;
		_enchantmentVfxOverride = GetNode<TextureRect>("%EnchantmentVfxOverride");
		CardHighlight = GetNode<NCardHighlight>("%Highlight");
		Body = GetNode<Control>("%CardContainer");
		_overlayContainer = GetNode("%OverlayContainer");
		_defaultEnchantmentPosition = _enchantmentTab.Position;
		Reload();
	}

	public override void _EnterTree()
	{
		SubscribeToModel(Model);
	}

	public override void _ExitTree()
	{
		UnsubscribeFromModel(Model);
	}

	private void SubscribeToModel(CardModel? model)
	{
		if (model != null && IsInsideTree())
		{
			model.AfflictionChanged += OnAfflictionChanged;
			model.EnchantmentChanged += OnEnchantmentChanged;
			SubscribeToEnchantment(model.Enchantment);
		}
	}

	private void UnsubscribeFromModel(CardModel? model)
	{
		if (model != null)
		{
			model.AfflictionChanged -= OnAfflictionChanged;
			model.EnchantmentChanged -= OnEnchantmentChanged;
			UnsubscribeFromEnchantment(model.Enchantment);
		}
	}

	private void SubscribeToEnchantment(EnchantmentModel? model)
	{
		if (model != null && IsInsideTree())
		{
			if (_subscribedEnchantment != null)
			{
				throw new InvalidOperationException($"Attempted to subscribe to enchantment {model}, but {this} is already subscribed to {_subscribedEnchantment}!");
			}
			_subscribedEnchantment = model;
			_subscribedEnchantment.StatusChanged += OnEnchantmentStatusChanged;
		}
	}

	private void UnsubscribeFromEnchantment(EnchantmentModel? model)
	{
		if (model != null && model == _subscribedEnchantment)
		{
			_subscribedEnchantment.StatusChanged -= OnEnchantmentStatusChanged;
			_subscribedEnchantment = null;
		}
	}

	public static void InitPool()
	{
		NodePool.Init<NCard>("res://scenes/cards/card.tscn", 30);
	}

	public static NCard? Create(CardModel card, ModelVisibility visibility = ModelVisibility.Visible)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCard nCard = NodePool.Get<NCard>();
		nCard.Model = card;
		nCard.Visibility = visibility;
		return nCard;
	}

	public static NCard? FindOnTable(CardModel card, PileType? overridePile = null)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		if (!CombatManager.Instance.IsInProgress)
		{
			return null;
		}
		NCombatUi nCombatUi = NCombatRoom.Instance?.Ui;
		if (nCombatUi == null)
		{
			return null;
		}
		CardPile? pile = card.Pile;
		return ((pile != null) ? new PileType?(pile.Type) : overridePile) switch
		{
			PileType.None => null, 
			PileType.Draw => null, 
			PileType.Hand => nCombatUi.Hand.GetCard(card) ?? nCombatUi.PlayQueue.GetCardNode(card) ?? nCombatUi.GetCardFromPlayContainer(card), 
			PileType.Discard => null, 
			PileType.Exhaust => null, 
			PileType.Play => nCombatUi.GetCardFromPlayContainer(card), 
			PileType.Deck => null, 
			null => null, 
			_ => throw new ArgumentOutOfRangeException(), 
		};
	}

	public Vector2 GetCurrentSize()
	{
		return defaultSize * base.Scale;
	}

	public void SetPreviewTarget(Creature? creature)
	{
		if (_previewTarget != creature)
		{
			_previewTarget = creature;
			UpdateVisuals(DisplayingPile, CardPreviewMode.Normal);
		}
	}

	public void UpdateVisuals(PileType pileType, CardPreviewMode previewMode)
	{
		if (!IsNodeReady())
		{
			return;
		}
		if (Model == null)
		{
			throw new InvalidOperationException("Cannot update text with no model.");
		}
		DisplayingPile = pileType;
		Creature target = _previewTarget ?? Model.CurrentTarget;
		UpdateTitleLabel();
		UpdateEnergyCostVisuals(pileType);
		UpdateStarCostVisuals(pileType);
		UpdateEnchantmentVisuals();
		Model.DynamicVars.ClearPreview();
		if (!_forceUnpoweredPreview)
		{
			Model.UpdateDynamicVarPreview(previewMode, target, Model.DynamicVars);
			if (Model.Enchantment != null)
			{
				Model.Enchantment.DynamicVars.ClearPreview();
				Model.UpdateDynamicVarPreview(previewMode, target, Model.Enchantment.DynamicVars);
			}
		}
		string text = ((previewMode != CardPreviewMode.Upgrade) ? Model.GetDescriptionForPile(pileType, target) : Model.GetDescriptionForUpgradePreview());
		switch (Visibility)
		{
		case ModelVisibility.Visible:
			_descriptionLabel.SetTextAutoSize("[center]" + text + "[/center]");
			break;
		case ModelVisibility.NotSeen:
			_descriptionLabel.SetTextAutoSize("[center][font_size=40]" + _unknownDescription.GetFormattedText() + "[/font_size][/center]");
			break;
		case ModelVisibility.Locked:
			_descriptionLabel.SetTextAutoSize("[center][font_size=40]" + _lockedDescription.GetFormattedText() + "[/font_size][/center]");
			break;
		default:
			throw new InvalidOperationException();
		}
	}

	public void ShowUpgradePreview()
	{
		UpdateVisuals(DisplayingPile, CardPreviewMode.Upgrade);
	}

	private void UpdateEnchantmentVisuals()
	{
		if (Model == null)
		{
			throw new InvalidOperationException("Cannot show enchantment with no model.");
		}
		EnchantmentModel enchantment = Model.Enchantment;
		if (enchantment != null)
		{
			_enchantmentTab.Visible = true;
			_enchantmentIcon.Texture = enchantment.Icon;
			_enchantmentLabel.SetTextAutoSize(enchantment.DisplayAmount.ToString());
			_enchantmentLabel.Visible = enchantment.ShowAmount;
			SetEnchantmentStatus(enchantment.Status);
		}
		else
		{
			_enchantmentTab.Visible = false;
		}
		if (Model.HasStarCostX || Model.CurrentStarCost >= 0)
		{
			_enchantmentTab.Position = _defaultEnchantmentPosition;
		}
		else
		{
			_enchantmentTab.Position = _defaultEnchantmentPosition + Vector2.Up * 45f;
		}
	}

	private void OnEnchantmentStatusChanged()
	{
		SetEnchantmentStatus(Model?.Enchantment?.Status ?? EnchantmentStatus.Disabled);
	}

	private void SetEnchantmentStatus(EnchantmentStatus status)
	{
		if (status == EnchantmentStatus.Disabled)
		{
			_enchantmentTab.Modulate = new Color(1f, 1f, 1f, 0.9f);
			ShaderMaterial shaderMaterial = (ShaderMaterial)_enchantmentTab.Material;
			shaderMaterial.SetShaderParameter(_h, 0.25);
			shaderMaterial.SetShaderParameter(_s, 0.1);
			shaderMaterial.SetShaderParameter(_v, 0.6);
			_enchantmentIcon.UseParentMaterial = true;
			_enchantmentLabel.SelfModulate = StsColors.gray;
		}
		else
		{
			_enchantmentTab.Modulate = Colors.White;
			ShaderMaterial shaderMaterial2 = (ShaderMaterial)_enchantmentTab.Material;
			shaderMaterial2.SetShaderParameter(_h, 0.25);
			shaderMaterial2.SetShaderParameter(_s, 0.4);
			shaderMaterial2.SetShaderParameter(_v, 0.6);
			_enchantmentIcon.UseParentMaterial = false;
			_enchantmentLabel.SelfModulate = Colors.White;
		}
	}

	private void UpdateEnergyCostVisuals(PileType pileType)
	{
		if (Visibility != ModelVisibility.Visible)
		{
			_energyLabel.SetTextAutoSize("?");
			_energyIcon.Visible = true;
			_energyLabel.AddThemeColorOverride(ThemeConstants.Label.fontColor, StsColors.cream);
			_energyLabel.AddThemeColorOverride(ThemeConstants.Label.fontOutlineColor, Model.Pool.EnergyOutlineColor);
			return;
		}
		if (Model.EnergyCost.CostsX)
		{
			_energyLabel.SetTextAutoSize("X");
			_energyIcon.Visible = true;
		}
		else
		{
			int withModifiers = Model.EnergyCost.GetWithModifiers(CostModifiers.All);
			_energyLabel.SetTextAutoSize(withModifiers.ToString());
			_energyIcon.Visible = withModifiers >= 0;
		}
		UpdateEnergyCostColor(pileType);
		if (pileType == PileType.Hand && !Model.CanPlay(out UnplayableReason reason, out AbstractModel _))
		{
			_unplayableEnergyIcon.Visible = !reason.HasResourceCostReason();
		}
		else
		{
			_unplayableEnergyIcon.Visible = false;
		}
	}

	public void SetPretendCardCanBePlayed(bool pretendCardCanBePlayed)
	{
		_pretendCardCanBePlayed = pretendCardCanBePlayed;
		UpdateEnergyCostVisuals(DisplayingPile);
		UpdateStarCostVisuals(DisplayingPile);
	}

	public void SetForceUnpoweredPreview(bool forceUnpoweredPreview)
	{
		_forceUnpoweredPreview = forceUnpoweredPreview;
	}

	private void UpdateEnergyCostColor(PileType pileType)
	{
		Color color = StsColors.cream;
		Color color2 = Model.Pool.EnergyOutlineColor;
		CardEnergyCost energyCost = Model.EnergyCost;
		if (energyCost != null && !energyCost.CostsX && energyCost.WasJustUpgraded)
		{
			color = StsColors.green;
			color2 = StsColors.energyGreenOutline;
		}
		else if (pileType == PileType.Hand)
		{
			CardCostColor energyCostColor = CardCostHelper.GetEnergyCostColor(Model, Model.CombatState);
			color = GetCostTextColorInHand(energyCostColor, _pretendCardCanBePlayed, color);
			color2 = GetCostOutlineColorInHand(energyCostColor, _pretendCardCanBePlayed, color2);
		}
		_energyLabel.AddThemeColorOverride(ThemeConstants.Label.fontColor, color);
		_energyLabel.AddThemeColorOverride(ThemeConstants.Label.fontOutlineColor, color2);
	}

	private void UpdateStarCostVisuals(PileType pileType)
	{
		if (Visibility != ModelVisibility.Visible)
		{
			_starLabel.SetTextAutoSize(string.Empty);
			_starIcon.Visible = false;
			_starLabel.AddThemeColorOverride(ThemeConstants.Label.fontColor, StsColors.cream);
			_starLabel.AddThemeColorOverride(ThemeConstants.Label.fontOutlineColor, Model.Pool.EnergyOutlineColor);
			return;
		}
		if (Model.HasStarCostX)
		{
			_starLabel.SetTextAutoSize("X");
			_starIcon.Visible = true;
		}
		else
		{
			_starLabel.SetTextAutoSize(Model.GetStarCostWithModifiers().ToString());
			_starIcon.Visible = Model.GetStarCostWithModifiers() >= 0;
		}
		UpdateStarCostColor(pileType);
		if (pileType == PileType.Hand && !Model.CanPlay(out UnplayableReason reason, out AbstractModel _))
		{
			_unplayableStarIcon.Visible = !reason.HasResourceCostReason();
		}
		else
		{
			_unplayableStarIcon.Visible = false;
		}
	}

	private void UpdateStarCostText(int cost)
	{
		if (Model.HasStarCostX)
		{
			_starLabel.SetTextAutoSize("X");
			_starIcon.Visible = true;
		}
		else if (cost >= 0)
		{
			_starLabel.SetTextAutoSize(cost.ToString());
			_starIcon.Visible = true;
		}
		else
		{
			_starIcon.Visible = false;
		}
	}

	private void UpdateStarCostColor(PileType pileType)
	{
		Color color = StsColors.cream;
		Color color2 = StsColors.defaultStarCostOutline;
		if (!Model.HasStarCostX && Model.WasStarCostJustUpgraded)
		{
			color = StsColors.green;
			color2 = StsColors.energyGreenOutline;
		}
		else if (pileType == PileType.Hand)
		{
			CardCostColor starCostColor = CardCostHelper.GetStarCostColor(Model, Model.CombatState);
			color = GetCostTextColorInHand(starCostColor, _pretendCardCanBePlayed, color);
			color2 = GetCostOutlineColorInHand(starCostColor, _pretendCardCanBePlayed, color2);
		}
		_starLabel.AddThemeColorOverride(ThemeConstants.Label.fontColor, color);
		_starLabel.AddThemeColorOverride(ThemeConstants.Label.fontOutlineColor, color2);
	}

	private static Color GetCostTextColorInHand(CardCostColor costColor, bool pretendCardCanBePlayed, Color defaultColor)
	{
		return costColor switch
		{
			CardCostColor.Unmodified => defaultColor, 
			CardCostColor.Increased => StsColors.energyBlue, 
			CardCostColor.Decreased => StsColors.green, 
			CardCostColor.InsufficientResources => pretendCardCanBePlayed ? defaultColor : StsColors.red, 
			_ => throw new ArgumentOutOfRangeException("costColor", costColor, null), 
		};
	}

	private static Color GetCostOutlineColorInHand(CardCostColor costColor, bool pretendCardCanBePlayed, Color defaultColor)
	{
		return costColor switch
		{
			CardCostColor.Unmodified => defaultColor, 
			CardCostColor.Increased => StsColors.energyBlueOutline, 
			CardCostColor.Decreased => StsColors.energyGreenOutline, 
			CardCostColor.InsufficientResources => pretendCardCanBePlayed ? defaultColor : StsColors.unplayableEnergyCostOutline, 
			_ => throw new ArgumentOutOfRangeException("costColor", costColor, null), 
		};
	}

	public void PlayRandomizeCostAnim()
	{
		RandomizeCostTween?.Kill();
		RandomizeCostTween = CreateTween();
		float offset = Rng.Chaotic.NextFloat(10f);
		RandomizeCostTween.TweenMethod(Callable.From(delegate(float t)
		{
			int num = (int)(offset + t) % 8;
			if (num > 3)
			{
				_energyLabel.SetTextAutoSize("?");
			}
			else
			{
				_energyLabel.SetTextAutoSize((t % 4f).ToString());
			}
		}), 0, 50, Rng.Chaotic.NextFloat(0.4f, 0.6f)).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Sine);
		RandomizeCostTween.Connect(Tween.SignalName.Finished, Callable.From(delegate
		{
			UpdateEnergyCostVisuals(Model.Pile.Type);
		}), 4u);
	}

	private void Reload()
	{
		if (!IsNodeReady() || Model == null)
		{
			return;
		}
		if (OS.HasFeature("editor"))
		{
			base.Name = $"{typeof(NCard)}-{Model.Id}";
		}
		_energyIcon.Texture = Model.EnergyIcon;
		UpdateTypePlaque();
		bool flag = Model.Rarity == CardRarity.Ancient;
		_portraitBorder.Visible = !flag;
		_portrait.Visible = !flag;
		_frame.Visible = !flag;
		_ancientPortrait.Visible = flag;
		_ancientBorder.Visible = flag;
		_ancientTextBg.Visible = flag;
		_ancientBanner.Visible = flag;
		_banner.Visible = !flag;
		_lock.Visible = Visibility == ModelVisibility.Locked;
		Texture2D portrait = Model.Portrait;
		if (Visibility != ModelVisibility.Visible)
		{
			if (_portraitBlurMaterial == null)
			{
				_portraitBlurMaterial = PreloadManager.Cache.GetMaterial(_portraitBlurMaterialPath);
			}
			if (flag)
			{
				if (_canvasGroupMaskBlurMaterial == null)
				{
					_canvasGroupMaskBlurMaterial = PreloadManager.Cache.GetMaterial(_canvasGroupMaskBlurMaterialPath);
				}
				_portraitCanvasGroup.Material = _canvasGroupMaskBlurMaterial;
			}
			else
			{
				if (_canvasGroupBlurMaterial == null)
				{
					_canvasGroupBlurMaterial = PreloadManager.Cache.GetMaterial(_canvasGroupBlurMaterialPath);
				}
				_portraitCanvasGroup.Material = _canvasGroupBlurMaterial;
			}
			_portrait.Material = _portraitBlurMaterial;
			_ancientPortrait.Material = _portraitBlurMaterial;
		}
		else
		{
			if (flag)
			{
				if (_canvasGroupMaskMaterial == null)
				{
					_canvasGroupMaskMaterial = PreloadManager.Cache.GetMaterial(_canvasGroupMaskMaterialPath);
				}
				_portraitCanvasGroup.Material = _canvasGroupMaskMaterial;
			}
			else
			{
				_portraitCanvasGroup.Material = null;
			}
			_portrait.Material = null;
			_ancientPortrait.Material = null;
		}
		if (Model.Rarity != CardRarity.Ancient)
		{
			_portrait.Texture = portrait;
			_portraitBorder.Texture = Model.PortraitBorder;
			_portraitBorder.Material = Model.BannerMaterial;
			_frame.Texture = Model.Frame;
			_banner.Material = Model.BannerMaterial;
			_banner.Texture = Model.BannerTexture;
		}
		else
		{
			_ancientTextBg.Texture = Model.AncientTextBg;
			_ancientPortrait.Texture = portrait;
			_banner.Material = null;
		}
		_frame.Material = Model.FrameMaterial;
		ReloadOverlay();
	}

	private void UpdateTypePlaque()
	{
		_typeLabel.SetTextAutoSize(Model.Type.ToLocString().GetFormattedText());
		Material bannerMaterial = Model.BannerMaterial;
		if (_typePlaque.Material != bannerMaterial)
		{
			_typePlaque.Material = Model.BannerMaterial;
		}
		Callable.From(UpdateTypePlaqueSizeAndPosition).CallDeferred();
	}

	private void UpdateTypePlaqueSizeAndPosition()
	{
		float num = _typePlaque.Position.X + _typePlaque.Size.X * 0.5f;
		NinePatchRect typePlaque = _typePlaque;
		Vector2 size = _typePlaque.Size;
		size.X = Mathf.Max(_typeLabel.Size.X + _typePlaqueXMargin, _typePlaqueMinXSize);
		typePlaque.Size = size;
		NinePatchRect typePlaque2 = _typePlaque;
		size = _typePlaque.Position;
		size.X = num - _typePlaque.Size.X * 0.5f;
		typePlaque2.Position = size;
	}

	private void UpdateTitleLabel()
	{
		string textAutoSize;
		Color color;
		Color color2;
		if (Visibility == ModelVisibility.NotSeen)
		{
			textAutoSize = _unknownTitle.GetFormattedText();
			color = StsColors.cream;
			color2 = GetTitleLabelOutlineColor();
		}
		else if (Visibility == ModelVisibility.Locked)
		{
			textAutoSize = _lockedTitle.GetFormattedText();
			color = StsColors.cream;
			color2 = GetTitleLabelOutlineColor();
		}
		else if (Model.CurrentUpgradeLevel == 0)
		{
			textAutoSize = Model.Title;
			color = StsColors.cream;
			color2 = GetTitleLabelOutlineColor();
		}
		else
		{
			textAutoSize = Model.Title;
			color = StsColors.green;
			color2 = StsColors.cardTitleOutlineSpecial;
		}
		_titleLabel.SetTextAutoSize(textAutoSize);
		_titleLabel.AddThemeColorOverride(ThemeConstants.Label.fontColor, color);
		_titleLabel.AddThemeColorOverride(ThemeConstants.Label.fontOutlineColor, color2);
	}

	private Color GetTitleLabelOutlineColor()
	{
		switch (_model.Rarity)
		{
		case CardRarity.None:
		case CardRarity.Basic:
		case CardRarity.Common:
		case CardRarity.Token:
			return StsColors.cardTitleOutlineCommon;
		case CardRarity.Uncommon:
			return StsColors.cardTitleOutlineUncommon;
		case CardRarity.Rare:
			return StsColors.cardTitleOutlineRare;
		case CardRarity.Curse:
			return StsColors.cardTitleOutlineCurse;
		case CardRarity.Quest:
			return StsColors.cardTitleOutlineQuest;
		case CardRarity.Status:
			return StsColors.cardTitleOutlineStatus;
		case CardRarity.Event:
			return StsColors.cardTitleOutlineSpecial;
		case CardRarity.Ancient:
			return StsColors.cardTitleOutlineCommon;
		default:
			throw new ArgumentOutOfRangeException();
		}
	}

	private void ReloadOverlay()
	{
		if (_cardOverlay != null)
		{
			_overlayContainer.RemoveChildSafely(_cardOverlay);
			_cardOverlay.QueueFreeSafely();
			_cardOverlay = null;
		}
		if (Model != null)
		{
			if (Model.Rarity == CardRarity.Ancient)
			{
				_frame.Visible = false;
				_ancientBorder.Visible = true;
				_ancientHighlight.Visible = true;
			}
			AfflictionModel affliction = Model.Affliction;
			if (affliction != null && affliction.HasOverlay)
			{
				_cardOverlay = Model.Affliction.CreateOverlay();
			}
			else if (Model.HasBuiltInOverlay)
			{
				_cardOverlay = Model.CreateOverlay();
			}
			if (_cardOverlay != null)
			{
				_overlayContainer.AddChildSafely(_cardOverlay);
			}
		}
	}

	private void OnAfflictionChanged()
	{
		ReloadOverlay();
	}

	private void OnEnchantmentChanged()
	{
		UnsubscribeFromEnchantment(_subscribedEnchantment);
		SubscribeToEnchantment(Model?.Enchantment);
		UpdateEnchantmentVisuals();
	}

	private string GetTitleText()
	{
		return _titleLabel.Text;
	}

	public void ActivateRewardScreenGlow()
	{
		if (_model.Rarity == CardRarity.Rare)
		{
			_sparkles.Visible = true;
			_rareGlow = NCardRareGlow.Create();
			if (_rareGlow != null)
			{
				Body.AddChildSafely(_rareGlow);
				Body.MoveChild(_rareGlow, 1);
			}
			CardHighlight.Modulate = NCardHighlight.gold;
		}
		else if (_model.Rarity == CardRarity.Uncommon)
		{
			_uncommonGlow = NCardUncommonGlow.Create();
			if (_uncommonGlow != null)
			{
				Body.AddChildSafely(_uncommonGlow);
				Body.MoveChild(_uncommonGlow, 1);
			}
			CardHighlight.Modulate = NCardHighlight.playableColor;
		}
	}

	public void FlashRelicOnCard(RelicModel relic)
	{
		NRelicFlashVfx nRelicFlashVfx = NRelicFlashVfx.Create(relic);
		Body.AddChildSafely(nRelicFlashVfx);
		nRelicFlashVfx.Scale = Vector2.One * 2f;
		nRelicFlashVfx.Position = base.Size * 0.5f;
		_flashVfx.Add(nRelicFlashVfx);
	}

	public void KillRarityGlow()
	{
		_rareGlow?.Kill();
		_uncommonGlow?.Kill();
	}

	public async Task AnimMultiCardPlay()
	{
		if (GodotObject.IsInstanceValid(this))
		{
			Vector2 scale = base.Scale;
			float y = base.Position.Y;
			PlayPileTween?.FastForwardToCompletion();
			PlayPileTween = CreateTween().SetParallel();
			PlayPileTween.TweenProperty(this, "modulate", StsColors.transparentBlack, 0.2);
			PlayPileTween.Chain();
			PlayPileTween.TweenInterval((SaveManager.Instance.PrefsSave.FastMode == FastModeType.Fast) ? 0.1 : 0.2);
			PlayPileTween.Chain();
			PlayPileTween.TweenProperty(this, "modulate", Colors.White, 0.2).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
			PlayPileTween.TweenProperty(this, "scale", scale, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic)
				.From(scale * 0.5f);
			PlayPileTween.TweenProperty(this, "position:y", y, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back)
				.From(y + 250f);
			await Cmd.CustomScaledWait(0.4f, 0.5f);
		}
	}

	public void AnimCardToPlayPile()
	{
		Vector2 targetPosition = PileType.Play.GetTargetPosition(this);
		PlayPileTween?.FastForwardToCompletion();
		PlayPileTween = CreateTween().SetParallel();
		PlayPileTween.TweenProperty(this, "position", targetPosition, 0.2).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		PlayPileTween.TweenProperty(this, "scale", Vector2.One * 0.8f, 0.2).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
	}

	public void OnReturnedFromPool()
	{
		if (IsNodeReady())
		{
			base.Position = Vector2.Zero;
			base.Rotation = 0f;
			base.Scale = Vector2.One;
			base.Modulate = Colors.White;
			base.Visible = true;
			Body.Visible = true;
			Body.Modulate = Colors.White;
			Body.Scale = Vector2.One;
			_visibility = ModelVisibility.Visible;
			CardHighlight.Modulate = NCardHighlight.playableColor;
			CardHighlight.AnimHideInstantly();
			_sparkles.Visible = false;
			_enchantmentTab.Visible = false;
			_enchantmentVfxOverride.Visible = false;
			_model = null;
			_previewTarget = null;
			_pretendCardCanBePlayed = false;
			_forceUnpoweredPreview = false;
			_portrait.Material = null;
			_ancientPortrait.Material = null;
			_portraitCanvasGroup.Material = null;
			DisplayingPile = PileType.None;
			this.ModelChanged = null;
		}
	}

	public void OnFreedToPool()
	{
		_rareGlow?.QueueFreeSafely();
		_rareGlow = null;
		_uncommonGlow?.QueueFreeSafely();
		_uncommonGlow = null;
		_cardOverlay?.QueueFreeSafely();
		_cardOverlay = null;
		_portrait.Texture = null;
		_enchantmentVfxOverride.Texture = null;
		foreach (NRelicFlashVfx item in _flashVfx)
		{
			if (item.IsValid())
			{
				item.QueueFreeSafely();
			}
		}
		_flashVfx.Clear();
		PlayPileTween?.Kill();
		PlayPileTween = null;
		RandomizeCostTween?.Kill();
		RandomizeCostTween = null;
	}
}
