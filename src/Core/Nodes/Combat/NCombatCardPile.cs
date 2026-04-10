using System;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.Capstones;
using MegaCrit.Sts2.Core.Nodes.Vfx;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public abstract partial class NCombatCardPile : NButton
{
	private CardPile? _pile;

	private Player? _localPlayer;

	private MegaLabel _countLabel;

	private Control _icon;

	private HoverTip _hoverTip;

	protected LocString _emptyPileMessage;

	private Tween? _bumpTween;

	private int _currentCount;

	private static readonly Vector2 _hoverScale = Vector2.One * 1.25f;

	private const double _unhoverAnimDur = 0.5;

	private const double _pressDownDur = 0.25;

	private static readonly Color _downColor = Colors.DarkGray;

	private Tween? _positionTween;

	private const double _animDuration = 0.5;

	protected Vector2 _showPosition = new Vector2(100f, 828f);

	protected Vector2 _hidePosition = new Vector2(-160f, 860f);

	protected abstract PileType Pile { get; }

	public override void _Ready()
	{
		if (GetType() != typeof(NCombatCardPile))
		{
			Log.Error($"{GetType()}");
			throw new InvalidOperationException("Don't call base._Ready()! Call ConnectSignals() instead.");
		}
		ConnectSignals();
	}

	protected override void ConnectSignals()
	{
		base.ConnectSignals();
		_icon = GetNode<Control>("Icon");
		_countLabel = GetNode<MegaLabel>("CountContainer/Count");
		SetAnimInOutPositions();
	}

	public override void _EnterTree()
	{
		base._EnterTree();
		if (_pile != null)
		{
			_pile.CardAddFinished -= AddCard;
			_pile.CardAddFinished += AddCard;
			_pile.CardRemoveFinished -= RemoveCard;
			_pile.CardRemoveFinished += RemoveCard;
		}
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		if (_pile != null)
		{
			_pile.CardAddFinished -= AddCard;
			_pile.CardRemoveFinished -= RemoveCard;
		}
		_positionTween?.Kill();
		_bumpTween?.Kill();
	}

	public virtual void Initialize(Player player)
	{
		_localPlayer = player;
		_pile = Pile.GetPile(_localPlayer);
		_pile.CardAddFinished += AddCard;
		_pile.CardRemoveFinished += RemoveCard;
		_currentCount = _pile.Cards.Count;
		_countLabel.SetTextAutoSize(_currentCount.ToString());
		_hoverTip = _pile.Type switch
		{
			PileType.Draw => new HoverTip(new LocString("static_hover_tips", "DRAW_PILE.title"), new LocString("static_hover_tips", "DRAW_PILE.description")), 
			PileType.Discard => new HoverTip(new LocString("static_hover_tips", "DISCARD_PILE.title"), new LocString("static_hover_tips", "DISCARD_PILE.description")), 
			PileType.Exhaust => new HoverTip(new LocString("static_hover_tips", "EXHAUST_PILE.title"), new LocString("static_hover_tips", "EXHAUST_PILE.description")), 
			_ => throw new ArgumentOutOfRangeException(), 
		};
	}

	protected virtual void SetAnimInOutPositions()
	{
	}

	protected override void OnRelease()
	{
		_bumpTween?.Kill();
		_bumpTween = CreateTween();
		_bumpTween.TweenProperty(_icon, "scale", base.IsFocused ? _hoverScale : Vector2.One, 0.05);
		_bumpTween.TweenProperty(_icon, "modulate", Colors.White, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		if (_pile == null || _localPlayer == null || !CombatManager.Instance.IsInProgress)
		{
			return;
		}
		if (_pile.IsEmpty)
		{
			NCapstoneContainer? instance = NCapstoneContainer.Instance;
			if (instance != null && instance.InUse)
			{
				NCapstoneContainer.Instance.Close();
			}
			NThoughtBubbleVfx child = NThoughtBubbleVfx.Create(_emptyPileMessage.GetFormattedText(), _localPlayer.Creature, 2.0);
			NCombatRoom.Instance?.CombatVfxContainer.AddChildSafely(child);
		}
		else if (NCapstoneContainer.Instance?.CurrentCapstoneScreen is NCardPileScreen nCardPileScreen && nCardPileScreen.Pile == _pile)
		{
			NCapstoneContainer.Instance.Close();
		}
		else
		{
			NCardPileScreen.ShowScreen(_pile, Hotkeys);
		}
	}

	protected override void OnFocus()
	{
		if (_pile != null)
		{
			NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, _hoverTip);
			nHoverTipSet.GlobalPosition = _pile.Type switch
			{
				PileType.Draw => base.GlobalPosition + new Vector2(14f, -375f), 
				PileType.Discard => base.GlobalPosition + new Vector2(-320f, -370f), 
				PileType.Exhaust => base.GlobalPosition + new Vector2(-320f, -125f), 
				_ => NHoverTipSet.CreateAndShow(this, _hoverTip).GlobalPosition, 
			};
			_bumpTween?.Kill();
			_bumpTween = CreateTween();
			_bumpTween.TweenProperty(_icon, "scale", _hoverScale, 0.05);
		}
	}

	protected override void OnUnfocus()
	{
		NHoverTipSet.Remove(this);
		_bumpTween?.Kill();
		_bumpTween = CreateTween().SetParallel();
		_bumpTween.TweenProperty(_icon, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_bumpTween.TweenProperty(_icon, "modulate", Colors.White, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	protected override void OnPress()
	{
		_bumpTween?.Kill();
		_bumpTween = CreateTween().SetParallel();
		_bumpTween.TweenProperty(_icon, "scale", Vector2.One, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		_bumpTween.TweenProperty(_icon, "modulate", _downColor, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
	}

	protected virtual void AddCard()
	{
		if (_pile != null)
		{
			_currentCount = Math.Min(_currentCount + 1, _pile.Cards.Count);
			_countLabel.SetTextAutoSize(_currentCount.ToString());
			_countLabel.PivotOffset = _countLabel.Size * 0.5f;
			_bumpTween?.Kill();
			_bumpTween = CreateTween().SetParallel();
			_icon.Scale = _hoverScale;
			_bumpTween.TweenProperty(_icon, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			_countLabel.Scale = _hoverScale;
			_bumpTween.TweenProperty(_countLabel, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		}
	}

	private void RemoveCard()
	{
		if (_pile != null)
		{
			_currentCount = Math.Max(_currentCount - 1, _pile.Cards.Count);
			_countLabel.SetTextAutoSize(_currentCount.ToString());
			_countLabel.PivotOffset = _countLabel.Size * 0.5f;
		}
	}

	public virtual void AnimIn()
	{
		base.Position = _hidePosition;
		_positionTween?.Kill();
		_positionTween = CreateTween();
		_positionTween.TweenProperty(this, "position", _showPosition, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	public void AnimOut()
	{
		base.Position = _showPosition;
		_positionTween?.Kill();
		_positionTween = CreateTween();
		_positionTween.TweenProperty(this, "position", _hidePosition, 0.5).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Back);
	}
}
