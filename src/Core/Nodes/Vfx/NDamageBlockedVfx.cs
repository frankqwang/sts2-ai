using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NDamageBlockedVfx : Node2D
{
	private static readonly LocString _blockedLoc = new LocString("vfx", "BLOCKED");

	private MegaLabel _label;

	private Tween? _tween;

	private static string ScenePath => SceneHelper.GetScenePath("vfx/vfx_blocked_text");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	public static NDamageBlockedVfx? Create(Creature target)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCreature creatureNode = NCombatRoom.Instance.GetCreatureNode(target);
		if (!creatureNode.IsInteractable)
		{
			return null;
		}
		NDamageBlockedVfx nDamageBlockedVfx = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NDamageBlockedVfx>(PackedScene.GenEditState.Disabled);
		nDamageBlockedVfx.GlobalPosition = creatureNode.VfxSpawnPosition + new Vector2(Rng.Chaotic.NextFloat(-20f, 20f), Rng.Chaotic.NextFloat(-60f, -40f));
		nDamageBlockedVfx.RotationDegrees = Rng.Chaotic.NextFloat(-2f, 2f);
		return nDamageBlockedVfx;
	}

	public override void _Ready()
	{
		_label = GetNode<MegaLabel>("Label");
		_label.SetTextAutoSize(_blockedLoc.GetRawText());
		TaskHelper.RunSafely(BlockAnim());
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
	}

	private async Task BlockAnim()
	{
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "scale", Vector2.One * 0.6f, 2.0).From(Vector2.One).SetEase(Tween.EaseType.Out)
			.SetTrans(Tween.TransitionType.Quad);
		_tween.TweenProperty(this, "position:y", base.Position.Y - 250f, 2.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Quad);
		_tween.TweenProperty(_label, "modulate", Colors.White, 2.0).From(new Color("21C0FF")).SetEase(Tween.EaseType.Out)
			.SetTrans(Tween.TransitionType.Cubic);
		_tween.TweenProperty(this, "modulate:a", 0f, 1.5).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Sine)
			.SetDelay(0.5);
		await ToSignal(_tween, Tween.SignalName.Finished);
		this.QueueFreeSafely();
	}
}
