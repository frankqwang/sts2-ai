using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes;

public partial class NActBanner : Control
{
	private MegaLabel _actNumber;

	private MegaLabel _actName;

	private ColorRect _banner;

	private static readonly string _path = SceneHelper.GetScenePath("ui/act_banner");

	private ActModel _act;

	private int _actIndex;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_path);

	public static NActBanner? Create(ActModel act, int actIndex)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NActBanner nActBanner = PreloadManager.Cache.GetScene(_path).Instantiate<NActBanner>(PackedScene.GenEditState.Disabled);
		nActBanner._act = act;
		nActBanner._actIndex = actIndex;
		return nActBanner;
	}

	public override void _Ready()
	{
		_actNumber = GetNode<MegaLabel>("ActNumber");
		_actName = GetNode<MegaLabel>("ActName");
		_banner = GetNode<ColorRect>("%Banner");
		LocString locString = new LocString("gameplay_ui", "ACT_NUMBER");
		locString.Add("actNumber", _actIndex + 1);
		_actNumber.SetTextAutoSize(locString.GetFormattedText());
		_actName.SetTextAutoSize(_act.Title.GetFormattedText());
		TaskHelper.RunSafely(AnimateVfx());
	}

	private async Task AnimateVfx()
	{
		_banner.Modulate = StsColors.transparentBlack;
		_actName.Modulate = StsColors.transparentWhite;
		_actNumber.Modulate = StsColors.transparentWhite;
		Tween tween = CreateTween().SetParallel();
		tween.TweenProperty(_banner, "modulate:a", 0.25f, 0.5).SetDelay(0.5);
		tween.TweenProperty(_actName, "modulate:a", 1f, 1.0).SetDelay(0.25);
		tween.TweenProperty(_actNumber, "modulate:a", 1f, 1.0).SetDelay(0.5);
		tween.TweenProperty(_actNumber, "position:y", 440f, 1.25).SetDelay(0.5).SetEase(Tween.EaseType.Out)
			.SetTrans(Tween.TransitionType.Quad)
			.From(450f);
		tween.Chain();
		tween.TweenInterval((SaveManager.Instance.PrefsSave.FastMode == FastModeType.Fast) ? 0.5 : 2.0);
		tween.Chain();
		tween.TweenProperty(this, "modulate:a", 0f, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Quad);
		await ToSignal(tween, Tween.SignalName.Finished);
		this.QueueFreeSafely();
	}
}
