using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Timeline;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NGainEpochVfx : Node
{
	private static int _vfxCount;

	private Control _epoch;

	private EpochModel _model;

	private TextureRect _portrait;

	private MegaLabel _label;

	private Tween? _tween;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	private static string ScenePath => SceneHelper.GetScenePath("vfx/vfx_gain_epoch");

	public override void _Ready()
	{
		_epoch = GetNode<Control>("%EpochContainer");
		_portrait = GetNode<TextureRect>("%Portrait");
		_portrait.Texture = _model.Portrait;
		_label = GetNode<MegaLabel>("%Label");
		_label.SetTextAutoSize(new LocString("vfx", "EPOCH_GAIN").GetRawText());
		TaskHelper.RunSafely(AnimateVfx());
	}

	private async Task AnimateVfx()
	{
		if (_vfxCount > 1)
		{
			int num = 3000 * (_vfxCount - 1);
			Log.Info($"Delaying Gain Epoch Vfx by: {num}ms");
			await Task.Delay(num);
		}
		_epoch.RotationDegrees = -30f;
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_epoch, "position:x", 164f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back);
		_tween.TweenProperty(_epoch, "rotation", 0f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back);
		_tween.Chain();
		_tween.TweenInterval(1.5);
		_tween.Chain();
		_tween.TweenProperty(this, "modulate:a", 0f, 1.0);
		await ToSignal(_tween, Tween.SignalName.Finished);
		this.QueueFreeSafely();
	}

	public static NGainEpochVfx Create(EpochModel model)
	{
		_vfxCount++;
		NGainEpochVfx nGainEpochVfx = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NGainEpochVfx>(PackedScene.GenEditState.Disabled);
		nGainEpochVfx._model = model;
		return nGainEpochVfx;
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
		_vfxCount--;
	}
}
