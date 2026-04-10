using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Relics;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens;

public partial class NChooseARelicSelection : Control, IOverlayScreen, IScreenContext
{
	private const float _relicXSpacing = 200f;

	private NCommonBanner _banner;

	private Control _relicRow;

	private NChoiceSelectionSkipButton _skipButton;

	private readonly TaskCompletionSource<IEnumerable<RelicModel>> _completionSource = new TaskCompletionSource<IEnumerable<RelicModel>>();

	private bool _screenComplete;

	private bool _relicSelected;

	private Tween? _cardTween;

	private Tween? _fadeTween;

	private IReadOnlyList<RelicModel> _relics;

	public NetScreenType ScreenType => NetScreenType.Rewards;

	private static string ScenePath => SceneHelper.GetScenePath("screens/choose_a_relic_selection_screen");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	public bool UseSharedBackstop => true;

	public Control DefaultFocusedControl
	{
		get
		{
			List<NRelicBasicHolder> list = _relicRow.GetChildren().OfType<NRelicBasicHolder>().ToList();
			return list[list.Count / 2];
		}
	}

	public static NChooseARelicSelection? ShowScreen(IReadOnlyList<RelicModel> relics)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NChooseARelicSelection nChooseARelicSelection = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NChooseARelicSelection>(PackedScene.GenEditState.Disabled);
		nChooseARelicSelection.Name = "NChooseACardSelectionScreen";
		nChooseARelicSelection._relics = relics;
		NOverlayStack.Instance.Push(nChooseARelicSelection);
		return nChooseARelicSelection;
	}

	public override void _Ready()
	{
		_banner = GetNode<NCommonBanner>("Banner");
		_banner.label.SetTextAutoSize(new LocString("gameplay_ui", "CHOOSE_RELIC_HEADER").GetRawText());
		_banner.AnimateIn();
		_relicRow = GetNode<Control>("RelicRow");
		Vector2 vector = Vector2.Left * (_relics.Count - 1) * 200f * 0.5f;
		for (int i = 0; i < _relics.Count; i++)
		{
			RelicModel relic = _relics[i];
			NRelicBasicHolder holder = NRelicBasicHolder.Create(relic);
			holder.Scale = Vector2.One * 2f;
			_relicRow.AddChildSafely(holder);
			holder.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(delegate
			{
				SelectHolder(holder);
			}));
			_cardTween = CreateTween().SetParallel();
			_cardTween.TweenProperty(holder, "position", holder.Position + vector + Vector2.Right * 200f * i, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			_cardTween.TweenProperty(holder, "modulate", Colors.White, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic)
				.From(Colors.Black);
		}
		_skipButton = GetNode<NChoiceSelectionSkipButton>("SkipButton");
		_skipButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnSkipButtonReleased));
		_skipButton.AnimateIn();
		List<NRelicBasicHolder> list = _relicRow.GetChildren().OfType<NRelicBasicHolder>().ToList();
		NRelicBasicHolder nRelicBasicHolder = _relicRow.GetChildren().OfType<NRelicBasicHolder>().ToList()[list.Count / 2];
		_skipButton.FocusNeighborTop = nRelicBasicHolder.GetPath();
		_skipButton.FocusNeighborBottom = _skipButton.GetPath();
		_skipButton.FocusNeighborLeft = _skipButton.GetPath();
		_skipButton.FocusNeighborRight = _skipButton.GetPath();
		for (int num = 0; num < _relicRow.GetChildCount(); num++)
		{
			Control child = _relicRow.GetChild<Control>(num);
			child.FocusNeighborBottom = child.GetPath();
			child.FocusNeighborTop = child.GetPath();
			child.FocusNeighborLeft = ((num > 0) ? _relicRow.GetChild(num - 1).GetPath() : _relicRow.GetChild(_relicRow.GetChildCount() - 1).GetPath());
			child.FocusNeighborRight = ((num < _relicRow.GetChildCount() - 1) ? _relicRow.GetChild(num + 1).GetPath() : _relicRow.GetChild(0).GetPath());
		}
	}

	public override void _ExitTree()
	{
		if (!_completionSource.Task.IsCompleted)
		{
			_completionSource.SetCanceled();
		}
	}

	private void SelectHolder(NRelicBasicHolder relicHolder)
	{
		RelicModel model = relicHolder.Relic.Model;
		_screenComplete = true;
		_relicSelected = true;
		_completionSource.SetResult(new RelicModel[1] { model });
	}

	public async Task<IEnumerable<RelicModel>> RelicsSelected()
	{
		IEnumerable<RelicModel> result = await _completionSource.Task;
		NOverlayStack.Instance.Remove(this);
		return result;
	}

	private void OnSkipButtonReleased(NButton _)
	{
		_screenComplete = true;
		_completionSource.SetResult(Array.Empty<RelicModel>());
	}

	public void AfterOverlayOpened()
	{
		base.Modulate = Colors.Transparent;
		_fadeTween?.Kill();
		_fadeTween = CreateTween();
		_fadeTween.TweenProperty(this, "modulate:a", 1f, 0.2);
	}

	public void AfterOverlayClosed()
	{
		_fadeTween?.Kill();
		this.QueueFreeSafely();
	}

	public void AfterOverlayShown()
	{
		base.Visible = true;
	}

	public void AfterOverlayHidden()
	{
		base.Visible = false;
	}
}
