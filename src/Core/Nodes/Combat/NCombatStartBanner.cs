using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NCombatStartBanner : Control
{
	private ColorRect _colorRect;

	private MegaLabel _label;

	private static readonly string _scenePath = SceneHelper.GetScenePath("combat/combat_start_banner");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public static NCombatStartBanner? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		if (NCombatUi.IsDebugHideTextVfx)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NCombatStartBanner>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		_colorRect = GetNode<ColorRect>("ColorRect");
		_colorRect.Modulate = Colors.Transparent;
		_label = GetNode<MegaLabel>("Label");
		_label.SetTextAutoSize(new LocString("gameplay_ui", "BATTLE_START").GetFormattedText());
		_label.Modulate = Colors.Transparent;
		TaskHelper.RunSafely(AnimateVfx());
	}

	private async Task AnimateVfx()
	{
		NDebugAudioManager.Instance?.Play(Rng.Chaotic.NextItem(TmpSfx.BattleStart));
		Tween tween = CreateTween().SetParallel();
		tween.TweenInterval(0.3);
		tween.Chain();
		tween.TweenProperty(_colorRect, "modulate:a", 0.5f, 0.75).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		tween.TweenProperty(_colorRect, "scale", Vector2.One, 0.75).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
			.From(Vector2.One * 1.2f);
		tween.TweenProperty(_label, "modulate:a", 1f, 1.2999999523162842).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		tween.TweenProperty(_label, "scale", Vector2.One, 0.75).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
			.From(Vector2.One * 2f);
		tween.TweenProperty(_label, "modulate:a", 0f, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic)
			.SetDelay(1.2999999523162842);
		await ToSignal(tween, Tween.SignalName.Finished);
		GetParent().AddChildSafely(NPlayerTurnBanner.Create(1));
		tween = CreateTween();
		tween.TweenProperty(_colorRect, "modulate:a", 0f, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic)
			.SetDelay(1.5);
		await ToSignal(tween, Tween.SignalName.Finished);
		this.QueueFreeSafely();
	}
}
