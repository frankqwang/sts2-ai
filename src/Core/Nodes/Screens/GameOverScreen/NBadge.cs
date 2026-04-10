using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;

namespace MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;

public partial class NBadge : Control
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/game_over_screen/badge");

	private Tween? _tween;

	public static string[] AssetPaths => new string[1] { SceneHelper.GetScenePath("screens/game_over_screen/badge") };

	public static NBadge Create(string label, Texture2D? icon = null)
	{
		NBadge nBadge = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NBadge>(PackedScene.GenEditState.Disabled);
		nBadge.GetNode<MegaLabel>("Label").SetTextAutoSize(label);
		if (icon != null)
		{
			nBadge.GetNode<TextureRect>("%Icon").Texture = icon;
		}
		return nBadge;
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
