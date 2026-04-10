using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NPlayerTurnBanner : Control
{
	private MegaLabel _label;

	private MegaLabel _turnLabel;

	private int _roundNumber;

	private static readonly string _scenePath = SceneHelper.GetScenePath("combat/player_turn_banner");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public static NPlayerTurnBanner? Create(int roundNumber)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		if (NCombatUi.IsDebugHideTextVfx)
		{
			return null;
		}
		NPlayerTurnBanner nPlayerTurnBanner = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NPlayerTurnBanner>(PackedScene.GenEditState.Disabled);
		nPlayerTurnBanner._roundNumber = roundNumber;
		return nPlayerTurnBanner;
	}

	public override void _Ready()
	{
		_label = GetNode<MegaLabel>("Label");
		if (CombatManager.Instance.PlayersTakingExtraTurn.Count > 0)
		{
			_label.SetTextAutoSize(new LocString("gameplay_ui", "PLAYER_TURN_EXTRA").GetFormattedText());
		}
		else
		{
			_label.SetTextAutoSize(new LocString("gameplay_ui", "PLAYER_TURN").GetFormattedText());
		}
		_turnLabel = GetNode<MegaLabel>("TurnNumber");
		LocString locString = new LocString("gameplay_ui", "TURN_COUNT");
		locString.Add("turnNumber", _roundNumber);
		_turnLabel.SetTextAutoSize(locString.GetFormattedText());
		base.Modulate = Colors.Transparent;
		TaskHelper.RunSafely(Display());
	}

	private async Task Display()
	{
		NDebugAudioManager.Instance?.Play("player_turn.mp3");
		Tween tween = CreateTween();
		tween.SetParallel();
		tween.TweenProperty(this, "modulate:a", 1f, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		tween.TweenProperty(_label, "position", _label.Position + new Vector2(0f, -50f), 1.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		tween.TweenProperty(_turnLabel, "position", _turnLabel.Position + new Vector2(0f, 50f), 1.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		await ToSignal(tween, Tween.SignalName.Finished);
		tween = CreateTween();
		tween.TweenInterval(0.4);
		tween.TweenProperty(this, "modulate:a", 0f, 0.30000001192092896).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Sine);
		await ToSignal(tween, Tween.SignalName.Finished);
		this.QueueFreeSafely();
	}
}
