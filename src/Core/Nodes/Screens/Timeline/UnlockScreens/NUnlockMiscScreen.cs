using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Timeline.UnlockScreens;

public partial class NUnlockMiscScreen : NUnlockScreen
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("timeline_screen/unlock_misc_screen");

	private MegaRichTextLabel _label;

	private string _textToSet;

	private Tween? _tween;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public static NUnlockMiscScreen Create()
	{
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NUnlockMiscScreen>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		ConnectSignals();
		_label = GetNode<MegaRichTextLabel>("%Label");
		_label.Text = _textToSet;
		_label.Modulate = StsColors.transparentBlack;
	}

	public override void Open()
	{
		base.Open();
		SfxCmd.Play("event:/sfx/ui/timeline/ui_timeline_unlock");
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_label, "modulate", Colors.White, 1.0).SetDelay(0.25);
	}

	public void SetUnlocks(string text)
	{
		_textToSet = text;
	}

	protected override void OnScreenPreClose()
	{
		_tween?.Kill();
	}

	protected override void OnScreenClose()
	{
		NTimelineScreen.Instance.EnableInput();
	}
}
