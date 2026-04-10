using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using Godot.Collections;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.RichTextTags;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes;

public partial class NAncientNameBanner : Control
{
	private MegaRichTextLabel _titleLabel;

	private RichTextAncientBanner _ancientBannerEffect;

	private MegaLabel _epithetLabel;

	private static readonly string _path = SceneHelper.GetScenePath("ui/ancient_name_banner");

	private AncientEventModel _ancient;

	private Tween? _moveTween;

	private Tween? _tween;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_path);

	public static NAncientNameBanner? Create(AncientEventModel ancient)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NAncientNameBanner nAncientNameBanner = PreloadManager.Cache.GetScene(_path).Instantiate<NAncientNameBanner>(PackedScene.GenEditState.Disabled);
		nAncientNameBanner._ancient = ancient;
		return nAncientNameBanner;
	}

	public override void _Ready()
	{
		_titleLabel = GetNode<MegaRichTextLabel>("%Title");
		string text = _ancient.Title.GetFormattedText().ToUpper();
		_ancientBannerEffect = new RichTextAncientBanner();
		_ancientBannerEffect.CenterCharacter = GetTextCenterGlyphIndex(text, _titleLabel.GetThemeFont(ThemeConstants.RichTextLabel.normalFont, "RichTextLabel"), _titleLabel.GetThemeFontSize(ThemeConstants.RichTextLabel.normalFontSize, "RichTextLabel"));
		_titleLabel.InstallEffect(_ancientBannerEffect);
		_titleLabel.BbcodeEnabled = true;
		_titleLabel.Text = "[ancient_banner]" + text + "[/ancient_banner]";
		_epithetLabel = GetNode<MegaLabel>("%Epithet");
		_epithetLabel.SetTextAutoSize(_ancient.Epithet.GetFormattedText());
		TaskHelper.RunSafely(AnimateVfx());
	}

	private async Task AnimateVfx()
	{
		_epithetLabel.Position = new Vector2(0f, 18f);
		_epithetLabel.Modulate = Colors.Transparent;
		_moveTween = CreateTween();
		_moveTween.TweenProperty(this, "position:y", -100f, 4.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Circ);
		_tween = CreateTween().SetParallel();
		_tween.TweenMethod(Callable.From<float>(UpdateGlyphSpace), 1f, 0f, 3.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenMethod(Callable.From<float>(UpdateTransform), 0f, 1f, 3.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Spring);
		_tween.TweenProperty(_epithetLabel, "position:y", 42f, 2.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic)
			.SetDelay(1.0);
		_tween.TweenProperty(_epithetLabel, "modulate:a", 1f, 1.0).SetDelay(1.5);
		_tween.Chain();
		_tween.TweenInterval(1.5);
		_tween.Chain();
		_tween.TweenProperty(_titleLabel, "modulate", Colors.Red, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(_titleLabel, "modulate:a", 0f, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		_tween.TweenProperty(_epithetLabel, "modulate", Colors.Red, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(_epithetLabel, "modulate:a", 0f, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		await ToSignal(_tween, Tween.SignalName.Finished);
		_moveTween.Kill();
		_moveTween = CreateTween().SetParallel();
		base.Position = new Vector2(0f, -80f);
		_titleLabel.Modulate = Colors.White;
		_titleLabel.HorizontalAlignment = HorizontalAlignment.Left;
		_titleLabel.VerticalAlignment = VerticalAlignment.Bottom;
		_titleLabel.Position = Vector2.Zero;
		_titleLabel.AddThemeFontSizeOverride(ThemeConstants.RichTextLabel.normalFontSize, 54);
		_titleLabel.AddThemeColorOverride(ThemeConstants.RichTextLabel.fontOutlineColor, Colors.Transparent);
		_titleLabel.AddThemeColorOverride(ThemeConstants.RichTextLabel.fontShadowColor, Colors.Transparent);
		_titleLabel.AddThemeColorOverride(ThemeConstants.RichTextLabel.defaultColor, StsColors.cream);
		_epithetLabel.HorizontalAlignment = HorizontalAlignment.Left;
		_epithetLabel.VerticalAlignment = VerticalAlignment.Bottom;
		_epithetLabel.Modulate = new Color(1f, 1f, 1f, 0f);
		_epithetLabel.AddThemeFontSizeOverride(ThemeConstants.Label.fontSize, 18);
		_epithetLabel.AddThemeColorOverride(ThemeConstants.Label.fontOutlineColor, Colors.Transparent);
		_epithetLabel.AddThemeColorOverride(ThemeConstants.Label.fontShadowColor, Colors.Transparent);
		_epithetLabel.AddThemeColorOverride(ThemeConstants.Label.fontColor, StsColors.cream);
		_moveTween.TweenProperty(_epithetLabel, "modulate:a", 0.5f, 2.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Circ);
		_moveTween.TweenProperty(this, "position:x", 48f, 2.0).SetEase(Tween.EaseType.Out).From(0)
			.SetTrans(Tween.TransitionType.Circ);
	}

	private void UpdateTransform(float obj)
	{
		_ancientBannerEffect.Rotation = obj;
	}

	private void UpdateGlyphSpace(float spacing)
	{
		_ancientBannerEffect.Spacing = spacing * 1000f;
	}

	private float GetTextCenterGlyphIndex(string text, Font font, int fontSize)
	{
		using TextParagraph textParagraph = new TextParagraph();
		textParagraph.AddString(text, font, fontSize);
		float num = 0f;
		TextServer primaryInterface = TextServerManager.Singleton.GetPrimaryInterface();
		Array<Dictionary> array = primaryInterface.ShapedTextGetGlyphs(textParagraph.GetLineRid(0));
		foreach (Dictionary item in array)
		{
			float num2 = item.GetValueOrDefault("advance").AsSingle();
			num += num2;
		}
		float num3 = 0f;
		int num4 = 0;
		foreach (Dictionary item2 in array)
		{
			float num5 = item2.GetValueOrDefault("advance").AsSingle();
			num3 += num5;
			if (num3 > num * 0.5f)
			{
				return (float)num4 + (num * 0.5f - (num3 - num5)) / num5;
			}
			num4++;
		}
		return 0f;
	}

	public override void _ExitTree()
	{
		_moveTween?.Kill();
		_tween?.Kill();
		_titleLabel.RemoveThemeFontSizeOverride(ThemeConstants.RichTextLabel.normalFontSize);
		_titleLabel.RemoveThemeColorOverride(ThemeConstants.RichTextLabel.fontOutlineColor);
		_titleLabel.RemoveThemeColorOverride(ThemeConstants.RichTextLabel.fontShadowColor);
		_titleLabel.RemoveThemeColorOverride(ThemeConstants.RichTextLabel.defaultColor);
		_epithetLabel.RemoveThemeFontSizeOverride(ThemeConstants.Label.fontSize);
		_epithetLabel.RemoveThemeColorOverride(ThemeConstants.Label.fontOutlineColor);
		_epithetLabel.RemoveThemeColorOverride(ThemeConstants.Label.fontShadowColor);
		_epithetLabel.RemoveThemeColorOverride(ThemeConstants.Label.fontColor);
		base._ExitTree();
	}
}
