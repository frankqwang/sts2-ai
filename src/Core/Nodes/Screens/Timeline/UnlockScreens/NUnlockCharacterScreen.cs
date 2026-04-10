using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Timeline;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Timeline.UnlockScreens;

public partial class NUnlockCharacterScreen : NUnlockScreen
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("timeline_screen/unlock_character_screen");

	private MegaRichTextLabel _topLabel;

	private MegaRichTextLabel _bottomLabel;

	private Control _spineAnchor;

	private NCreatureVisuals _creatureVisuals;

	private GpuParticles2D _rareGlow;

	private EpochModel _epoch;

	private CharacterModel _character;

	private Tween? _tween;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public static NUnlockCharacterScreen Create(EpochModel epoch, CharacterModel character)
	{
		NUnlockCharacterScreen nUnlockCharacterScreen = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NUnlockCharacterScreen>(PackedScene.GenEditState.Disabled);
		nUnlockCharacterScreen._character = character;
		nUnlockCharacterScreen._epoch = epoch;
		return nUnlockCharacterScreen;
	}

	public override void _Ready()
	{
		ConnectSignals();
		_topLabel = GetNode<MegaRichTextLabel>("%TopLabel");
		_bottomLabel = GetNode<MegaRichTextLabel>("%BottomLabel");
		_spineAnchor = GetNode<Control>("%SpineAnchor");
		_rareGlow = GetNode<GpuParticles2D>("%RareGlow");
		_topLabel.Text = new LocString("epochs", _epoch.Id + ".unlock").GetFormattedText();
		_bottomLabel.Text = new LocString("epochs", _epoch.Id + ".unlockText").GetFormattedText();
		_topLabel.Modulate = StsColors.transparentBlack;
		_bottomLabel.Modulate = StsColors.transparentBlack;
		_spineAnchor.Modulate = StsColors.transparentBlack;
		_creatureVisuals = _character.CreateVisuals();
		_spineAnchor.AddChildSafely(_creatureVisuals);
	}

	public override void Open()
	{
		base.Open();
		SfxCmd.Play("event:/sfx/ui/timeline/ui_timeline_unlock");
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_spineAnchor, "modulate", Colors.White, 0.25).SetDelay(0.25);
		_tween.TweenProperty(_topLabel, "modulate", Colors.White, 1.0).SetDelay(1.0);
		_tween.TweenProperty(_bottomLabel, "modulate", Colors.White, 1.0).SetDelay(1.5);
		_tween.TweenProperty(_rareGlow, "modulate:a", 1f, 0.5).SetDelay(1.0);
		_creatureVisuals.SpineBody.GetAnimationState().AddAnimation("idle_loop");
		_creatureVisuals.SpineBody.GetAnimationState().AddAnimation("attack", 0.5f, loop: false);
		_creatureVisuals.SpineBody.GetAnimationState().AddAnimation("idle_loop");
	}

	protected override void OnScreenPreClose()
	{
		_tween?.Kill();
		_tween = CreateTween();
		_tween.TweenProperty(_rareGlow, "modulate:a", 0f, 0.5);
	}

	protected override void OnScreenClose()
	{
		NTimelineScreen.Instance.EnableInput();
	}
}
