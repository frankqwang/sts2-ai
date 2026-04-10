using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.Capstones;
using MegaCrit.Sts2.Core.Nodes.Screens.CardLibrary;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;

namespace MegaCrit.Sts2.Core.Nodes.Screens;

public partial class NCardPileScreen : Control, ICapstoneScreen, IScreenContext
{
	private ColorRect _background;

	private NCardGrid _grid;

	private NButton _backButton;

	private MegaRichTextLabel _bottomLabel;

	private Tween? _currentTween;

	private string[] _closeHotkeys = Array.Empty<string>();

	private static string ScenePath => SceneHelper.GetScenePath("/screens/card_pile_screen");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	public NetScreenType ScreenType => NetScreenType.CardPile;

	public CardPile Pile { get; private set; }

	public bool UseSharedBackstop => true;

	public Control? DefaultFocusedControl => _grid.DefaultFocusedControl;

	public override void _Ready()
	{
		_bottomLabel = GetNode<MegaRichTextLabel>("%BottomLabel");
		switch (Pile.Type)
		{
		case PileType.Draw:
			_bottomLabel.Text = "[center]" + new LocString("gameplay_ui", "DRAW_PILE_INFO").GetFormattedText();
			break;
		case PileType.Discard:
			_bottomLabel.Text = "[center]" + new LocString("gameplay_ui", "DISCARD_PILE_INFO").GetFormattedText();
			break;
		case PileType.Exhaust:
			_bottomLabel.Text = "[center]" + new LocString("gameplay_ui", "EXHAUST_PILE_INFO").GetFormattedText();
			break;
		default:
			_bottomLabel.Visible = false;
			Log.Info("CardPileScreen has no info text.");
			break;
		}
		_backButton = GetNode<NButton>("BackButton");
		_backButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnReturnButtonPressed));
		_backButton.Enable();
		_grid = GetNode<NCardGrid>("CardGrid");
		OnPileContentsChanged();
		_grid.InsetForTopBar();
		_background = GetNode<ColorRect>("Background");
		_background.Modulate = StsColors.transparentBlack;
		_currentTween = CreateTween();
		_currentTween.TweenProperty(_background, "modulate", StsColors.screenBackdrop, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		base.ProcessMode = (ProcessModeEnum)(base.Visible ? 0 : 4);
	}

	public override void _EnterTree()
	{
		base._EnterTree();
		Pile.ContentsChanged += OnPileContentsChanged;
		string[] closeHotkeys = _closeHotkeys;
		foreach (string hotkey in closeHotkeys)
		{
			NHotkeyManager.Instance.PushHotkeyReleasedBinding(hotkey, NCapstoneContainer.Instance.Close);
		}
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		Pile.ContentsChanged -= OnPileContentsChanged;
		string[] closeHotkeys = _closeHotkeys;
		foreach (string hotkey in closeHotkeys)
		{
			NHotkeyManager.Instance.RemoveHotkeyReleasedBinding(hotkey, NCapstoneContainer.Instance.Close);
		}
	}

	public static NCardPileScreen ShowScreen(CardPile pile, string[] closeHotkeys)
	{
		NDebugAudioManager.Instance?.Play("map_open.mp3");
		NCardPileScreen nCardPileScreen = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NCardPileScreen>(PackedScene.GenEditState.Disabled);
		nCardPileScreen.Name = $"{"NCardPileScreen"}-{pile.Type}";
		nCardPileScreen.Pile = pile;
		nCardPileScreen._closeHotkeys = closeHotkeys;
		NCapstoneContainer.Instance.Open(nCardPileScreen);
		return nCardPileScreen;
	}

	private void OnPileContentsChanged()
	{
		List<CardModel> list = Pile.Cards.ToList();
		if (Pile.Type == PileType.Draw)
		{
			list.Sort((CardModel c1, CardModel c2) => (c1.Rarity != c2.Rarity) ? c1.Rarity.CompareTo(c2.Rarity) : string.Compare(c1.Id.Entry, c2.Id.Entry, StringComparison.Ordinal));
		}
		NCardGrid grid = _grid;
		PileType type = Pile.Type;
		int num = 1;
		List<SortingOrders> list2 = new List<SortingOrders>(num);
		CollectionsMarshal.SetCount(list2, num);
		Span<SortingOrders> span = CollectionsMarshal.AsSpan(list2);
		int index = 0;
		span[index] = SortingOrders.Ascending;
		grid.SetCards(list, type, list2);
	}

	private void OnReturnButtonPressed(NButton _)
	{
		NCapstoneContainer.Instance.Close();
	}

	public void AfterCapstoneOpened()
	{
		base.Visible = true;
	}

	public void AfterCapstoneClosed()
	{
		base.Visible = false;
		this.QueueFreeSafely();
	}
}
