using System;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;

namespace MegaCrit.sts2.Core.Nodes.TopBar;

public partial class NTopBarGold : NClickableControl
{
	private static readonly HoverTip _hoverTip = new HoverTip(new LocString("static_hover_tips", "MONEY_POUCH.title"), new LocString("static_hover_tips", "MONEY_POUCH.description"));

	private Player? _player;

	private MegaLabel _goldLabel;

	private MegaLabel _goldPopupLabel;

	private int _currentGold;

	private int _additionalGold;

	private bool _alreadyRunning;

	private CancellationTokenSource? _animCts;

	public override void _Ready()
	{
		_goldLabel = GetNode<MegaLabel>("%GoldLabel");
		_goldPopupLabel = GetNode<MegaLabel>("%GoldPopup");
		_goldPopupLabel.Modulate = Colors.Transparent;
		ConnectSignals();
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		_animCts?.Cancel();
		_animCts?.Dispose();
		if (_player != null)
		{
			_player.GoldChanged -= UpdateGold;
		}
	}

	public void Initialize(Player player)
	{
		_player = player;
		_currentGold = _player.Gold;
		_goldLabel.SetTextAutoSize($"{_currentGold}");
		_player.GoldChanged += UpdateGold;
	}

	private void UpdateGold()
	{
		TaskHelper.RunSafely(UpdateGoldAnim());
	}

	private async Task UpdateGoldAnim()
	{
		if (_player == null)
		{
			return;
		}
		int currentGold = _player.Gold - _currentGold;
		_additionalGold = (_currentGold = currentGold);
		_currentGold = _player.Gold;
		_goldPopupLabel.SetTextAutoSize(((_additionalGold > 0) ? "+" : "") + _additionalGold);
		if (_alreadyRunning)
		{
			return;
		}
		_animCts?.Dispose();
		_animCts = new CancellationTokenSource();
		CancellationToken ct = _animCts.Token;
		_alreadyRunning = true;
		try
		{
			Tween tween = CreateTween().SetParallel();
			tween.TweenProperty(_goldPopupLabel, "modulate:a", 1f, 0.15000000596046448);
			tween.TweenProperty(_goldPopupLabel, "position:y", _goldPopupLabel.Position.Y + 30f, 0.25);
			await tween.AwaitFinished(ct);
			await Task.Delay(150, ct);
			while (_additionalGold != 0)
			{
				ct.ThrowIfCancellationRequested();
				int num = 1;
				if (Mathf.Abs(_additionalGold) > 100)
				{
					num = 75;
				}
				else if (Mathf.Abs(_additionalGold) > 50)
				{
					num = 10;
				}
				_additionalGold = ((_additionalGold > 0) ? (_additionalGold - num) : (_additionalGold + num));
				_goldPopupLabel.SetTextAutoSize(((_additionalGold >= 0) ? "+" : "") + _additionalGold);
				_goldLabel.SetTextAutoSize($"{_player.Gold - _additionalGold}");
				await Task.Delay((int)Mathf.Lerp(10f, 20f, Mathf.Max(0, 10 - Mathf.Abs(_additionalGold))), ct);
			}
			await Task.Delay(250, ct);
			Tween tween2 = CreateTween().SetParallel();
			tween2.TweenProperty(_goldPopupLabel, "modulate:a", 0f, 0.10000000149011612);
			tween2.TweenProperty(_goldPopupLabel, "position:y", _goldPopupLabel.Position.Y - 30f, 0.25).FromCurrent();
			_goldLabel.SetTextAutoSize($"{_player.Gold}");
		}
		catch (OperationCanceledException)
		{
		}
		finally
		{
			_alreadyRunning = false;
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
}
