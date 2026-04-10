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
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NStunnedVfx : Node2D
{
	private const string _scenePath = "res://scenes/vfx/stunned_vfx.tscn";

	private static LocString _stunnedLoc = new LocString("vfx", "STUNNED");

	private MegaLabel _label;

	private Creature _creature;

	private Tween? _textTween;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>("res://scenes/vfx/stunned_vfx.tscn");

	public override void _Ready()
	{
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(_creature);
		if (nCreature == null)
		{
			this.QueueFreeSafely();
			return;
		}
		_label = GetNode<MegaLabel>("%Label");
		base.GlobalPosition = nCreature.GetTopOfHitbox();
		TaskHelper.RunSafely(StartVfx());
	}

	public override void _ExitTree()
	{
		_textTween?.Kill();
	}

	private async Task StartVfx()
	{
		_label.SetTextAutoSize(_stunnedLoc.GetFormattedText());
		_textTween = CreateTween();
		_textTween.TweenProperty(_label, "modulate:a", 1f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_textTween.TweenInterval(0.5);
		_textTween.TweenProperty(_label, "modulate:a", 0f, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Linear);
		CreateTween().TweenProperty(_label, "position:y", _label.Position.Y - 100f, 2.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Quart);
		await ToSignal(_textTween, Tween.SignalName.Finished);
		this.QueueFreeSafely();
	}

	public static NStunnedVfx? Create(Creature creature)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		if (NCombatUi.IsDebugHideTextVfx)
		{
			return null;
		}
		NStunnedVfx nStunnedVfx = PreloadManager.Cache.GetScene("res://scenes/vfx/stunned_vfx.tscn").Instantiate<NStunnedVfx>(PackedScene.GenEditState.Disabled);
		nStunnedVfx._creature = creature;
		return nStunnedVfx;
	}
}
