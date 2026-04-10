using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Relics;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Timeline.UnlockScreens;

public partial class NUnlockRelicsScreen : NUnlockScreen
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("timeline_screen/unlock_relics_screen");

	private Control _relicRow;

	private NCommonBanner _banner;

	private IReadOnlyList<RelicModel> _relics;

	private Tween? _relicTween;

	private const float _relicXOffset = 350f;

	private static readonly Vector2 _relicScale = Vector2.One * 3f;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public static NUnlockRelicsScreen Create()
	{
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NUnlockRelicsScreen>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		ConnectSignals();
		_banner = GetNode<NCommonBanner>("%Banner");
		_banner.label.SetTextAutoSize(new LocString("timeline", "UNLOCK_RELICS_BANNER").GetRawText());
		_banner.AnimateIn();
		_relicRow = GetNode<Control>("%RelicRow");
		LocString locString = new LocString("timeline", "UNLOCK_RELICS");
		GetNode<MegaRichTextLabel>("%ExplanationText").Text = "[center]" + locString.GetFormattedText() + "[/center]";
	}

	public override void Open()
	{
		base.Open();
		SfxCmd.Play("event:/sfx/ui/timeline/ui_timeline_unlock");
		Vector2 vector = Vector2.Left * (_relics.Count - 1) * 350f * 0.5f;
		_relicTween = CreateTween().SetParallel();
		int num = 0;
		foreach (RelicModel relic in _relics)
		{
			NRelicBasicHolder nRelicBasicHolder = NRelicBasicHolder.Create(relic);
			_relicRow.AddChildSafely(nRelicBasicHolder);
			nRelicBasicHolder.Modulate = StsColors.transparentBlack;
			nRelicBasicHolder.Scale = _relicScale;
			_relicTween.TweenProperty(nRelicBasicHolder, "position", vector + Vector2.Right * 350f * num, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			_relicTween.TweenProperty(nRelicBasicHolder, "modulate", Colors.White, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
			num++;
		}
	}

	public void SetRelics(IReadOnlyList<RelicModel> relics)
	{
		_relics = relics;
	}

	protected override void OnScreenClose()
	{
		NTimelineScreen.Instance.EnableInput();
	}
}
