using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Bestiary;

public partial class NBestiaryMoveButton : NButton
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/bestiary/bestiary_move_button");

	private MegaLabel _label;

	private Tween? _tween;

	public BestiaryMonsterMove Move { get; private set; }

	public override void _Ready()
	{
		ConnectSignals();
		_label = GetNode<MegaLabel>("%Label");
		_label.SetTextAutoSize(Move.moveName);
		_label.PivotOffset = new Vector2(0f, _label.Size.Y * 0.5f);
	}

	public static NBestiaryMoveButton Create(BestiaryMonsterMove move)
	{
		NBestiaryMoveButton nBestiaryMoveButton = PreloadManager.Cache.GetAsset<PackedScene>(_scenePath).Instantiate<NBestiaryMoveButton>(PackedScene.GenEditState.Disabled);
		nBestiaryMoveButton.Move = move;
		return nBestiaryMoveButton;
	}

	protected override void OnPress()
	{
	}

	protected override void OnFocus()
	{
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_label, "scale", Vector2.One * 1.05f, 0.05);
	}

	protected override void OnUnfocus()
	{
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_label, "scale", Vector2.One, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
	}

	public void PlaySfx()
	{
		if (Move.sfx != null)
		{
			SfxCmd.Play(Move.sfx);
		}
	}
}
