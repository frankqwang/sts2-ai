using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.CommonUi;

public partial class NSaveIndicator : Control
{
	private Tween? _tween;

	public override void _Ready()
	{
		GetNode<MegaRichTextLabel>("Label").SetTextAutoSize(new LocString("gameplay_ui", "GAME_SAVED").GetFormattedText());
		base.Modulate = Colors.Transparent;
	}

	private void SavedGame()
	{
		if (!NCombatUi.IsDebugHideTextVfx)
		{
			_tween?.Kill();
			_tween = CreateTween();
			_tween.TweenInterval(0.5);
			_tween.TweenProperty(this, "modulate", Colors.White, 1.0);
			_tween.TweenInterval(0.5);
			_tween.TweenProperty(this, "modulate", Colors.Transparent, 1.0);
		}
	}

	public override void _EnterTree()
	{
		SaveManager.Instance.Saved += SavedGame;
	}

	public override void _ExitTree()
	{
		SaveManager.Instance.Saved -= SavedGame;
	}
}
