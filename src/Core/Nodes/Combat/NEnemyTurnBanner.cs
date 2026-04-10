using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NEnemyTurnBanner : Control
{
	private MegaLabel _label;

	private static readonly string _scenePath = SceneHelper.GetScenePath("combat/enemy_turn_banner");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public static NEnemyTurnBanner? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		if (NCombatUi.IsDebugHideTextVfx)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NEnemyTurnBanner>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		_label = GetNode<MegaLabel>("Label");
		_label.SetTextAutoSize(new LocString("gameplay_ui", "ENEMY_TURN").GetFormattedText());
		base.Modulate = Colors.Transparent;
		TaskHelper.RunSafely(Display());
	}

	private async Task Display()
	{
		NDebugAudioManager.Instance?.Play("enemy_turn.mp3");
		_label.Scale = Vector2.One * 2f;
		Tween tween = CreateTween();
		tween.SetParallel();
		tween.TweenProperty(_label, "scale", Vector2.One, 0.75).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		tween.TweenProperty(this, "modulate:a", 1f, 1.2999999523162842).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		tween.Chain();
		tween.TweenProperty(_label, "modulate", Colors.Red, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
			.FromCurrent();
		tween.TweenProperty(this, "modulate:a", 0f, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		await ToSignal(tween, Tween.SignalName.Finished);
		this.QueueFreeSafely();
	}
}
