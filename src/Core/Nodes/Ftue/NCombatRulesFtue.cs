using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Debug;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Ftue;

public partial class NCombatRulesFtue : NFtue
{
	[Export(PropertyHint.None, "")]
	private Texture2D _image1;

	[Export(PropertyHint.None, "")]
	private Texture2D _image2;

	[Export(PropertyHint.None, "")]
	private Texture2D _image3;

	public const string id = "combat_rules_ftue";

	private static readonly string _scenePath = SceneHelper.GetScenePath("ftue/combat_rules_ftue");

	private NButton _prevButton;

	private NButton _nextButton;

	private MegaLabel _pageCount;

	private TextureRect _image;

	private MegaRichTextLabel _bodyText;

	private MegaLabel _header;

	private int _currentPage = 1;

	private const int _totalPages = 3;

	private Vector2 _imagePosition;

	private Vector2 _textPosition;

	private Tween? _pageTurnTween;

	private const double _textTweenSpeed = 0.6;

	private static readonly Vector2 _imageAnimOffset = new Vector2(200f, 0f);

	public override void _Ready()
	{
		_image = GetNode<TextureRect>("Image");
		_bodyText = GetNode<MegaRichTextLabel>("%Description");
		_pageCount = GetNode<MegaLabel>("PageCount");
		_header = GetNode<MegaLabel>("Header");
		_prevButton = GetNode<NButton>("LeftArrow");
		_nextButton = GetNode<NButton>("RightArrow");
		_image.Modulate = Colors.Transparent;
		_bodyText.Modulate = Colors.Transparent;
		_prevButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(ToggleLeft));
		_nextButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(ToggleRight));
		_prevButton.Visible = false;
		_prevButton.Disable();
		_nextButton.Visible = false;
		_nextButton.Disable();
		_pageCount.Visible = false;
		_header.Visible = false;
	}

	public void Start()
	{
		NModalContainer.Instance?.ShowBackstop();
		_currentPage = 1;
		_imagePosition = _image.Position;
		_bodyText.Text = new LocString("ftues", "TUTORIAL_FTUE_BODY_1").GetFormattedText();
		_bodyText.Modulate = StsColors.transparentWhite;
		_textPosition = _bodyText.Position;
		LocString locString = new LocString("ftues", "COMBAT_BASICS_FTUE_PAGE_COUNT");
		locString.Add("totalPages", 3m);
		locString.Add("currentPage", _currentPage);
		_pageCount.SetTextAutoSize(locString.GetFormattedText());
		_header.SetTextAutoSize(new LocString("ftues", "COMBAT_BASICS_FTUE_HEADER").GetFormattedText());
		_nextButton.Visible = true;
		_nextButton.Enable();
		_pageCount.Visible = true;
		_header.Visible = true;
		_pageTurnTween = CreateTween().SetParallel();
		_pageTurnTween.TweenProperty(_image, "modulate:a", 1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic)
			.From(0f);
		_pageTurnTween.TweenProperty(_bodyText, "modulate:a", 1f, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Linear);
		_pageTurnTween.TweenProperty(_bodyText, "visible_ratio", 1f, 0.6).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Sine)
			.From(0f);
	}

	public static NCombatRulesFtue? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NCombatRulesFtue>(PackedScene.GenEditState.Disabled);
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
			if (inputEvent.IsActionPressed(MegaInput.left) && _prevButton.IsEnabled)
			{
				ToggleLeft(_prevButton);
			}
			if (inputEvent.IsActionPressed(MegaInput.right) && _nextButton.IsEnabled)
			{
				ToggleRight(_nextButton);
			}
		}
	}

	private void ToggleLeft(NButton _)
	{
		_currentPage--;
		switch (_currentPage)
		{
		case 1:
			_prevButton.Visible = false;
			_prevButton.Disable();
			_bodyText.SetTextAutoSize(new LocString("ftues", "TUTORIAL_FTUE_BODY_1").GetFormattedText());
			_image.Texture = _image1;
			break;
		case 2:
			_bodyText.SetTextAutoSize(new LocString("ftues", "TUTORIAL_FTUE_BODY_2").GetFormattedText());
			_image.Texture = _image2;
			break;
		}
		LocString locString = new LocString("ftues", "COMBAT_BASICS_FTUE_PAGE_COUNT");
		locString.Add("totalPages", 3m);
		locString.Add("currentPage", _currentPage);
		_pageCount.SetTextAutoSize(locString.GetFormattedText());
		_pageTurnTween?.Kill();
		_pageTurnTween = CreateTween().SetParallel();
		_pageTurnTween.TweenProperty(_image, "modulate:a", 1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic)
			.From(0.5f);
		_pageTurnTween.TweenProperty(_image, "position", _imagePosition, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
			.From(_imagePosition - _imageAnimOffset);
		_pageTurnTween.TweenProperty(_bodyText, "position", _textPosition, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
			.From(_textPosition - _imageAnimOffset);
		_pageTurnTween.TweenProperty(_bodyText, "modulate:a", 1f, 0.6).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Linear)
			.From(0f);
		_pageTurnTween.TweenProperty(_bodyText, "visible_ratio", 1f, 0.6).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Sine)
			.From(0f);
	}

	private void ToggleRight(NButton _)
	{
		if (_currentPage == 3)
		{
			_pageTurnTween?.Kill();
			SaveManager.Instance.MarkFtueAsComplete("combat_rules_ftue");
			NCombatRoom.Instance.AddChildSafely(NCombatStartBanner.Create());
			CloseFtue();
			return;
		}
		_currentPage++;
		switch (_currentPage)
		{
		case 2:
			_prevButton.Visible = true;
			_prevButton.Enable();
			_bodyText.SetTextAutoSize(new LocString("ftues", "TUTORIAL_FTUE_BODY_2").GetFormattedText());
			_image.Texture = _image2;
			break;
		case 3:
			_bodyText.SetTextAutoSize(new LocString("ftues", "TUTORIAL_FTUE_BODY_3").GetFormattedText());
			_image.Texture = _image3;
			break;
		}
		LocString locString = new LocString("ftues", "COMBAT_BASICS_FTUE_PAGE_COUNT");
		locString.Add("totalPages", 3m);
		locString.Add("currentPage", _currentPage);
		_pageCount.SetTextAutoSize(locString.GetFormattedText());
		_pageTurnTween?.Kill();
		_pageTurnTween = CreateTween().SetParallel();
		_pageTurnTween.TweenProperty(_image, "modulate:a", 1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic)
			.From(0.5f);
		_pageTurnTween.TweenProperty(_image, "position", _imagePosition, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
			.From(_imagePosition + _imageAnimOffset);
		_pageTurnTween.TweenProperty(_bodyText, "position", _textPosition, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
			.From(_textPosition + _imageAnimOffset);
		_pageTurnTween.TweenProperty(_bodyText, "modulate:a", 1f, 0.6).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Linear)
			.From(0f);
		_pageTurnTween.TweenProperty(_bodyText, "visible_ratio", 1f, 0.6).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Sine)
			.From(0f);
	}
}
