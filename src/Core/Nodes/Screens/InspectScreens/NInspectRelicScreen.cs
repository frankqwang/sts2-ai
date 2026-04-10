using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Entities.Relics;
using MegaCrit.Sts2.Core.Extensions;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Debug;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;
using MegaCrit.Sts2.Core.Unlocks;

namespace MegaCrit.Sts2.Core.Nodes.Screens.InspectScreens;

public partial class NInspectRelicScreen : Control, IScreenContext
{
	private static readonly StringName _v = new StringName("v");

	private static readonly StringName _s = new StringName("s");

	private static readonly StringName _h = new StringName("h");

	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/inspect_relic_screen/inspect_relic_screen");

	private Control _popup;

	private Control _backstop;

	private MegaLabel _nameLabel;

	private MegaLabel _rarityLabel;

	private MegaRichTextLabel _description;

	private MegaRichTextLabel _flavor;

	private TextureRect _relicImage;

	private ShaderMaterial _frameHsv;

	private NGoldArrowButton _leftButton;

	private NGoldArrowButton _rightButton;

	private Control _hoverTipRect;

	private Tween? _screenTween;

	private Tween? _popupTween;

	private Vector2 _popupPosition;

	private float _leftButtonX;

	private float _rightButtonX;

	private const double _arrowButtonDelay = 0.1;

	private IReadOnlyList<RelicModel> _relics;

	private int _index;

	private HashSet<RelicModel> _allUnlockedRelics = new HashSet<RelicModel>();

	public static string[] AssetPaths => new string[1] { _scenePath };

	public Control? DefaultFocusedControl => null;

	public static NInspectRelicScreen? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NInspectRelicScreen>(PackedScene.GenEditState.Disabled);
	}

	public void Open(IReadOnlyList<RelicModel> relics, RelicModel relic)
	{
		Log.Info($"Inspecting Relic: {relic.Title}");
		UnlockState unlockState = SaveManager.Instance.GenerateUnlockStateFromProgress();
		_allUnlockedRelics.Clear();
		_allUnlockedRelics.UnionWith(unlockState.Relics);
		_relics = relics.ToList();
		_index = relics.IndexOf(relic);
		SetRelic(_index);
		base.Visible = true;
		_popup.Modulate = StsColors.transparentBlack;
		_leftButton.Modulate = StsColors.transparentBlack;
		_rightButton.Modulate = StsColors.transparentBlack;
		_leftButton.Enable();
		_rightButton.Enable();
		_backstop.Visible = true;
		_backstop.MouseFilter = MouseFilterEnum.Stop;
		_leftButton.MouseFilter = MouseFilterEnum.Stop;
		_rightButton.MouseFilter = MouseFilterEnum.Stop;
		_screenTween?.Kill();
		_screenTween = CreateTween().SetParallel();
		_screenTween.TweenProperty(_backstop, "modulate:a", 0.9f, 0.25);
		_screenTween.TweenProperty(_leftButton, "position:x", _leftButtonX, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back)
			.From(_leftButtonX + 100f)
			.SetDelay(0.1);
		_screenTween.TweenProperty(_leftButton, "modulate", Colors.White, 0.25).SetDelay(0.1);
		_screenTween.TweenProperty(_rightButton, "position:x", _rightButtonX, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back)
			.From(_rightButtonX - 100f)
			.SetDelay(0.1);
		_screenTween.TweenProperty(_rightButton, "modulate", Colors.White, 0.25).SetDelay(0.1);
		_popupTween?.Kill();
		_popupTween = CreateTween().SetParallel();
		_popupTween.TweenProperty(_popup, "modulate", Colors.White, 0.25);
		_popupTween.TweenProperty(_popup, "position", _popupPosition, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic)
			.From(_popupPosition + new Vector2(0f, 200f));
		ActiveScreenContext.Instance.Update();
		NHotkeyManager.Instance.AddBlockingScreen(this);
		NHotkeyManager.Instance.PushHotkeyPressedBinding(MegaInput.cancel, Close);
		NHotkeyManager.Instance.PushHotkeyPressedBinding(MegaInput.pauseAndBack, Close);
	}

	public override void _Ready()
	{
		_popup = GetNode<Control>("%Popup");
		_backstop = GetNode<Control>("%Backstop");
		_nameLabel = GetNode<MegaLabel>("%RelicName");
		_rarityLabel = GetNode<MegaLabel>("%Rarity");
		_description = GetNode<MegaRichTextLabel>("%RelicDescription");
		_flavor = GetNode<MegaRichTextLabel>("%FlavorText");
		_relicImage = GetNode<TextureRect>("%RelicImage");
		_frameHsv = (ShaderMaterial)GetNode<Control>("%Frame").Material;
		_leftButton = GetNode<NGoldArrowButton>("%LeftArrow");
		_rightButton = GetNode<NGoldArrowButton>("%RightArrow");
		_popupPosition = _popup.Position;
		_hoverTipRect = GetNode<Control>("HoverTipRect");
		_backstop = GetNode<NButton>("Backstop");
		_backstop.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnBackstopPressed));
		_leftButton = GetNode<NGoldArrowButton>("LeftArrow");
		_leftButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnLeftButtonPressed));
		_rightButton = GetNode<NGoldArrowButton>("RightArrow");
		_rightButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnRightButtonPressed));
		_leftButtonX = _leftButton.Position.X;
		_rightButtonX = _rightButton.Position.X;
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (!IsVisibleInTree() || NDevConsole.Instance.Visible)
		{
			return;
		}
		Control control = GetViewport().GuiGetFocusOwner();
		if ((!(control is TextEdit) && !(control is LineEdit)) || 1 == 0)
		{
			if (inputEvent.IsActionPressed(MegaInput.left))
			{
				OnLeftButtonPressed(_leftButton);
			}
			if (inputEvent.IsActionPressed(MegaInput.right))
			{
				OnRightButtonPressed(_rightButton);
			}
		}
	}

	private void OnRightButtonPressed(NButton button)
	{
		SetRelic(_index + 1);
		_popup.Modulate = Colors.White;
		_popupTween?.Kill();
		_popupTween = CreateTween().SetParallel();
		_popupTween.TweenProperty(_popup, "position", _popupPosition, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
			.From(_popupPosition + new Vector2(100f, 0f));
	}

	private void OnLeftButtonPressed(NButton button)
	{
		SetRelic(_index - 1);
		_popup.Modulate = Colors.White;
		_popupTween?.Kill();
		_popupTween = CreateTween().SetParallel();
		_popupTween.TweenProperty(_popup, "position", _popupPosition, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
			.From(_popupPosition + new Vector2(-100f, 0f));
	}

	private void SetRelic(int index)
	{
		_index = Math.Clamp(index, 0, _relics.Count - 1);
		_leftButton.Visible = _index > 0;
		_leftButton.MouseFilter = (MouseFilterEnum)((_index > 0) ? 0 : 2);
		_rightButton.Visible = _index < _relics.Count - 1;
		_rightButton.MouseFilter = (MouseFilterEnum)((_index < _relics.Count - 1) ? 0 : 2);
		UpdateRelicDisplay();
	}

	private void UpdateRelicDisplay()
	{
		RelicModel relicModel = _relics[_index];
		if (!_allUnlockedRelics.Contains(relicModel.CanonicalInstance))
		{
			_nameLabel.SetTextAutoSize(new LocString("inspect_relic_screen", "LOCKED_TITLE").GetFormattedText());
			_rarityLabel.SetTextAutoSize(string.Empty);
			_relicImage.SelfModulate = Colors.White;
			_description.SetTextAutoSize(new LocString("inspect_relic_screen", "LOCKED_DESCRIPTION").GetFormattedText());
			_flavor.Text = string.Empty;
			SetRarityVisuals(RelicRarity.Common);
			_relicImage.Texture = PreloadManager.Cache.GetTexture2D(ImageHelper.GetImagePath("packed/common_ui/locked_model.png"));
		}
		else if (!SaveManager.Instance.IsRelicSeen(relicModel))
		{
			_nameLabel.SetTextAutoSize(new LocString("inspect_relic_screen", "UNDISCOVERED_TITLE").GetFormattedText());
			_rarityLabel.SetTextAutoSize(string.Empty);
			_relicImage.SelfModulate = StsColors.ninetyPercentBlack;
			_description.SetTextAutoSize(new LocString("inspect_relic_screen", "UNDISCOVERED_DESCRIPTION").GetFormattedText());
			_flavor.Text = string.Empty;
			SetRarityVisuals(relicModel.Rarity);
			_relicImage.Texture = relicModel.BigIcon;
		}
		else
		{
			_nameLabel.SetTextAutoSize(relicModel.Title.GetFormattedText());
			LocString locString = new LocString("gameplay_ui", "RELIC_RARITY." + relicModel.Rarity.ToString().ToUpperInvariant());
			_rarityLabel.SetTextAutoSize(locString.GetFormattedText());
			_relicImage.SelfModulate = Colors.White;
			_description.SetTextAutoSize(relicModel.DynamicDescription.GetFormattedText());
			_flavor.SetTextAutoSize(relicModel.Flavor.GetFormattedText());
			SetRarityVisuals(relicModel.Rarity);
			_relicImage.Texture = relicModel.BigIcon;
		}
		NHoverTipSet.Clear();
		if (SaveManager.Instance.IsRelicSeen(relicModel))
		{
			NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, relicModel.HoverTipsExcludingRelic);
			nHoverTipSet.SetAlignment(_hoverTipRect, HoverTip.GetHoverTipAlignment(this));
		}
	}

	private void SetRarityVisuals(RelicRarity rarity)
	{
		Vector3 vector;
		switch (rarity)
		{
		case RelicRarity.None:
		case RelicRarity.Starter:
		case RelicRarity.Common:
			_rarityLabel.Modulate = StsColors.cream;
			vector = new Vector3(0.95f, 0.25f, 0.9f);
			break;
		case RelicRarity.Uncommon:
			_rarityLabel.Modulate = StsColors.blue;
			vector = new Vector3(0.426f, 0.8f, 1.1f);
			break;
		case RelicRarity.Rare:
			_rarityLabel.Modulate = StsColors.gold;
			vector = new Vector3(1f, 0.8f, 1.15f);
			break;
		case RelicRarity.Shop:
			_rarityLabel.Modulate = StsColors.blue;
			vector = new Vector3(0.525f, 2.5f, 0.85f);
			break;
		case RelicRarity.Event:
			_rarityLabel.Modulate = StsColors.green;
			vector = new Vector3(0.23f, 0.75f, 0.9f);
			break;
		case RelicRarity.Ancient:
			_rarityLabel.Modulate = StsColors.red;
			vector = new Vector3(0.875f, 3f, 0.9f);
			break;
		default:
			Log.Error("Unspecified relic rarity: " + rarity);
			throw new ArgumentOutOfRangeException();
		}
		_frameHsv.SetShaderParameter(_h, vector.X);
		_frameHsv.SetShaderParameter(_s, vector.Y);
		_frameHsv.SetShaderParameter(_v, vector.Z);
	}

	public void Close()
	{
		if (base.Visible)
		{
			NHotkeyManager.Instance.RemoveBlockingScreen(this);
			NHotkeyManager.Instance.RemoveHotkeyPressedBinding(MegaInput.cancel, Close);
			NHotkeyManager.Instance.RemoveHotkeyPressedBinding(MegaInput.pauseAndBack, Close);
			_backstop.MouseFilter = MouseFilterEnum.Ignore;
			_leftButton.MouseFilter = MouseFilterEnum.Ignore;
			_rightButton.MouseFilter = MouseFilterEnum.Ignore;
			_leftButton.Disable();
			_rightButton.Disable();
			_screenTween?.Kill();
			_screenTween = CreateTween().SetParallel();
			_screenTween.TweenProperty(_backstop, "modulate:a", 0f, 0.25);
			_screenTween.TweenProperty(_leftButton, "modulate:a", 0f, 0.1);
			_screenTween.TweenProperty(_rightButton, "modulate:a", 0f, 0.1);
			_screenTween.TweenProperty(_popup, "modulate", StsColors.transparentWhite, 0.1);
			_screenTween.Chain().TweenCallback(Callable.From(delegate
			{
				base.Visible = false;
				ActiveScreenContext.Instance.Update();
			}));
			NHoverTipSet.Clear();
		}
	}

	private void OnBackstopPressed(NButton _)
	{
		Close();
	}
}
