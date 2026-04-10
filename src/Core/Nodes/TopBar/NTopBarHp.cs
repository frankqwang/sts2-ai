using System;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;

namespace MegaCrit.sts2.Core.Nodes.TopBar;

public partial class NTopBarHp : NClickableControl
{
	private static readonly HoverTip _hoverTip = new HoverTip(new LocString("static_hover_tips", "HIT_POINTS.title"), new LocString("static_hover_tips", "HIT_POINTS.description"));

	private Player? _player;

	private MegaLabel _hpLabel;

	public override void _Ready()
	{
		_hpLabel = GetNode<MegaLabel>("%HpLabel");
		ConnectSignals();
	}

	public override void _ExitTree()
	{
		if (_player != null)
		{
			_player.Creature.CurrentHpChanged -= UpdateHealth;
			_player.Creature.MaxHpChanged -= UpdateHealth;
		}
	}

	public void Initialize(Player player)
	{
		_player = player;
		_player.Creature.CurrentHpChanged += UpdateHealth;
		_player.Creature.MaxHpChanged += UpdateHealth;
		UpdateHealth(0, 0);
	}

	private void UpdateHealth(int _, int __)
	{
		if (_player != null)
		{
			Creature creature = _player.Creature;
			int currentHp = creature.CurrentHp;
			int maxHp = creature.MaxHp;
			_hpLabel.SetTextAutoSize($"{currentHp}/{maxHp}");
		}
	}

	protected override void OnFocus()
	{
		NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, _hoverTip);
		nHoverTipSet.GlobalPosition = base.GlobalPosition + new Vector2(0f, base.Size.Y + 20f);
	}

	protected override void OnUnfocus()
	{
		NHoverTipSet.Remove(this);
	}

	public async Task LerpAtNeow()
	{
		if (_player != null)
		{
			_hpLabel.SetTextAutoSize($"0/{_player.Creature.MaxHp}");
			await Cmd.Wait(0.5f);
			Tween tween = CreateTween();
			tween.TweenMethod(Callable.From<float>(UpdateHpTween), 0f, 1f, 1.0).SetEase(Tween.EaseType.InOut).SetTrans(Tween.TransitionType.Cubic);
		}
	}

	private void UpdateHpTween(float tweenAmount)
	{
		if (_player != null)
		{
			Creature creature = _player.Creature;
			int value = (int)Math.Round((float)creature.CurrentHp * tweenAmount);
			int maxHp = creature.MaxHp;
			_hpLabel.SetTextAutoSize($"{value}/{maxHp}");
		}
	}
}
