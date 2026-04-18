using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;

namespace MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;

public partial class NScoreLine : Control
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/game_over_screen/score_line");

	private Tween? _tween;

	public static NScoreLine Create(string label, string score, Texture2D? icon = null)
	{
		NScoreLine nScoreLine = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NScoreLine>(PackedScene.GenEditState.Disabled);
		nScoreLine.GetNode<MegaLabel>("%Label").SetTextAutoSize(label);
		nScoreLine.GetNode<MegaLabel>("%Score").SetTextAutoSize(score);
		if (icon != null)
		{
			nScoreLine.GetNode<TextureRect>("%Icon").Texture = icon;
		}
		return nScoreLine;
	}

	public async Task AnimateIn()
	{
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "modulate:a", 1f, 0.3);
		_tween.TweenProperty(this, "position:x", base.Position.X, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Spring)
			.From(base.Position.X - 50f);
		if (SaveManager.Instance.PrefsSave.FastMode != FastModeType.Instant)
		{
			_tween.Chain();
			_tween.TweenInterval(0.1);
		}
		await ToSignal(_tween, Tween.SignalName.Finished);
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
	}
}
