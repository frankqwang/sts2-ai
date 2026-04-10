using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Powers;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NPowerAppliedVfx : Control
{
	private const string _scenePath = "res://scenes/vfx/power_applied_vfx.tscn";

	private TextureRect _icon;

	private TextureRect _iconEcho;

	private MegaLabel _powerField;

	private PowerModel _power;

	private int _amount;

	private Tween? _textTween;

	private Tween? _spriteTween;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>("res://scenes/vfx/power_applied_vfx.tscn");

	public override void _Ready()
	{
		_icon = GetNode<TextureRect>("Icon");
		_iconEcho = GetNode<TextureRect>("Icon/IconEcho");
		_powerField = GetNode<MegaLabel>("Label");
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(_power.Owner);
		if (nCreature == null)
		{
			this.QueueFreeSafely();
			return;
		}
		base.GlobalPosition = nCreature.VfxSpawnPosition;
		TaskHelper.RunSafely(StartVfx());
	}

	public static NPowerAppliedVfx? Create(PowerModel power, int amount)
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
		NPowerAppliedVfx nPowerAppliedVfx = PreloadManager.Cache.GetScene("res://scenes/vfx/power_applied_vfx.tscn").Instantiate<NPowerAppliedVfx>(PackedScene.GenEditState.Disabled);
		nPowerAppliedVfx._power = power;
		nPowerAppliedVfx._amount = amount;
		return nPowerAppliedVfx;
	}

	public override void _ExitTree()
	{
		_spriteTween?.Kill();
		_textTween?.Kill();
	}

	private async Task StartVfx()
	{
		_powerField.SetTextAutoSize(_power.Title.GetFormattedText());
		_icon.Texture = _power.BigIcon;
		_iconEcho.Texture = _power.BigIcon;
		_powerField.Modulate = ((_power.GetTypeForAmount(_amount) == PowerType.Buff) ? StsColors.green : StsColors.red);
		_powerField.Position = new Vector2(_powerField.Position.X, _powerField.Position.Y - 200f);
		_spriteTween = CreateTween().SetParallel();
		_spriteTween.TweenProperty(_icon, "scale", Vector2.One * 0.8f, 1.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
			.From(Vector2.One * 0.4f);
		_spriteTween.TweenProperty(_icon, "modulate:a", 0.5f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Sine);
		_spriteTween.TweenProperty(_icon, "modulate:a", 0f, 1.0).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Sine)
			.SetDelay(0.25);
		_textTween = CreateTween().SetParallel();
		CreateTween().TweenProperty(_powerField, "position:y", _powerField.Position.Y + 50f, 1.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		_textTween.TweenProperty(_powerField, "modulate:a", 1f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_textTween.TweenProperty(_powerField, "modulate:a", 0f, 0.75).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Expo)
			.From(1f)
			.SetDelay(0.25);
		await ToSignal(_spriteTween, Tween.SignalName.Finished);
		this.QueueFreeSafely();
	}
}
