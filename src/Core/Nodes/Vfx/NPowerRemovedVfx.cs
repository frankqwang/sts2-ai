using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NPowerRemovedVfx : Node2D
{
	private const string _scenePath = "res://scenes/vfx/power_removed_vfx.tscn";

	private static LocString _wearsOffLoc = new LocString("vfx", "POWER_WEARS_OFF");

	private TextureRect _sprite;

	private MegaLabel _powerField;

	private Control _vfxContainer;

	private PowerModel _power;

	private Tween? _textTween;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>("res://scenes/vfx/power_removed_vfx.tscn");

	public override void _Ready()
	{
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(_power.Owner);
		if (nCreature == null)
		{
			this.QueueFreeSafely();
			return;
		}
		_sprite = GetNode<TextureRect>("%TextureRect");
		_powerField = GetNode<MegaLabel>("%PowerField");
		_vfxContainer = GetNode<Control>("%Container");
		base.GlobalPosition = nCreature.GetTopOfHitbox();
		TaskHelper.RunSafely(StartVfx());
	}

	public override void _ExitTree()
	{
		_textTween?.Kill();
	}

	private async Task StartVfx()
	{
		GetNode<MegaLabel>("%WearsOff").SetTextAutoSize(_wearsOffLoc.GetRawText());
		_powerField.SetTextAutoSize(_power.Title.GetFormattedText());
		_sprite.Texture = _power.BigIcon;
		_vfxContainer.Position -= _vfxContainer.Size * 0.5f;
		_textTween = CreateTween();
		_textTween.TweenProperty(_vfxContainer, "modulate:a", 1f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_textTween.TweenInterval(0.5);
		_textTween.TweenProperty(_vfxContainer, "modulate:a", 0f, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Linear);
		CreateTween().TweenProperty(_vfxContainer, "position:y", _powerField.Position.Y - 160f, 2.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Quart);
		await ToSignal(_textTween, Tween.SignalName.Finished);
		this.QueueFreeSafely();
	}

	public static NPowerRemovedVfx? Create(PowerModel power)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		if (NCombatUi.IsDebugHideTextVfx)
		{
			return null;
		}
		if (!power.ShouldPlayVfx)
		{
			return null;
		}
		NPowerRemovedVfx nPowerRemovedVfx = PreloadManager.Cache.GetScene("res://scenes/vfx/power_removed_vfx.tscn").Instantiate<NPowerRemovedVfx>(PackedScene.GenEditState.Disabled);
		nPowerRemovedVfx._power = power;
		return nPowerRemovedVfx;
	}
}
