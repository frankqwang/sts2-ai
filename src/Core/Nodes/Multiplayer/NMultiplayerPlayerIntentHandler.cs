using System;
using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Potions;
using MegaCrit.Sts2.Core.Nodes.Relics;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Multiplayer;

public partial class NMultiplayerPlayerIntentHandler : Control
{
	private const string _scenePath = "combat/multiplayer_player_intent";

	private NMultiplayerCardIntent _cardIntent;

	private NRelic _relicIntent;

	private NPotion _potionIntent;

	private NPower _powerIntent;

	private Control _hitbox;

	private NRemoteTargetingIndicator _targetingIndicator;

	private MegaRichTextLabel _cardThinkyDots;

	private MegaRichTextLabel _relicThinkyDots;

	private MegaRichTextLabel _potionThinkyDots;

	private MegaRichTextLabel _powerThinkyDots;

	private bool _shouldShowHoverTip;

	private Player _player;

	private AbstractModel? _displayedModel;

	private NHoverTipSet? _hoverTips;

	private bool _isInPlayerChoice;

	private NCard? _cardInPlayAwaitingPlayerChoice;

	private Tween? _tween;

	public NMultiplayerCardIntent CardIntent => _cardIntent;

	public static NMultiplayerPlayerIntentHandler? Create(Player player)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		if (RunManager.Instance.IsSinglePlayerOrFakeMultiplayer)
		{
			return null;
		}
		NMultiplayerPlayerIntentHandler nMultiplayerPlayerIntentHandler = PreloadManager.Cache.GetScene(SceneHelper.GetScenePath("combat/multiplayer_player_intent")).Instantiate<NMultiplayerPlayerIntentHandler>(PackedScene.GenEditState.Disabled);
		nMultiplayerPlayerIntentHandler._player = player;
		return nMultiplayerPlayerIntentHandler;
	}

	public override void _Ready()
	{
		_cardIntent = GetNode<NMultiplayerCardIntent>("%CardIntent");
		_relicIntent = GetNode<NRelic>("%RelicIntent");
		_potionIntent = GetNode<NPotion>("%PotionIntent");
		_powerIntent = GetNode<NPower>("%PowerIntent");
		_hitbox = GetNode<Control>("%Hitbox");
		_targetingIndicator = GetNode<NRemoteTargetingIndicator>("%TargetingIndicator");
		_cardThinkyDots = _cardIntent.GetNode<MegaRichTextLabel>("ThinkyDots");
		_relicThinkyDots = _relicIntent.GetNode<MegaRichTextLabel>("ThinkyDots");
		_potionThinkyDots = _potionIntent.GetNode<MegaRichTextLabel>("ThinkyDots");
		_powerThinkyDots = _powerIntent.GetNode<MegaRichTextLabel>("ThinkyDots");
		_targetingIndicator.Initialize(_player);
		_cardIntent.Visible = false;
		_relicIntent.Visible = false;
		_potionIntent.Visible = false;
		_powerIntent.Visible = false;
		HideThinkyDots();
		RunManager.Instance.ActionQueueSet.ActionEnqueued += OnActionEnqueued;
		if (!LocalContext.IsMe(_player))
		{
			_hitbox.Connect(Control.SignalName.FocusEntered, Callable.From(OnHitboxEntered));
			_hitbox.Connect(Control.SignalName.FocusExited, Callable.From(OnHitboxExited));
			_hitbox.Connect(Control.SignalName.MouseEntered, Callable.From(OnHitboxEntered));
			_hitbox.Connect(Control.SignalName.MouseExited, Callable.From(OnHitboxExited));
			RunManager.Instance.HoveredModelTracker.HoverChanged += OnHoverChanged;
			RunManager.Instance.InputSynchronizer.StateChanged += OnPeerInputStateChanged;
			RunManager.Instance.InputSynchronizer.StateRemoved += OnPeerInputStateRemoved;
		}
		base.ProcessMode = ProcessModeEnum.Disabled;
	}

	public override void _ExitTree()
	{
		RunManager.Instance.ActionQueueSet.ActionEnqueued -= OnActionEnqueued;
		if (!LocalContext.IsMe(_player))
		{
			RunManager.Instance.HoveredModelTracker.HoverChanged -= OnHoverChanged;
			RunManager.Instance.InputSynchronizer.StateChanged -= OnPeerInputStateChanged;
			RunManager.Instance.InputSynchronizer.StateRemoved -= OnPeerInputStateRemoved;
		}
	}

	private void OnHitboxEntered()
	{
		_shouldShowHoverTip = true;
		RefreshHoverTips();
	}

	private void OnHitboxExited()
	{
		_shouldShowHoverTip = false;
		RefreshHoverTips();
	}

	private void OnHoverChanged(ulong playerId)
	{
		if (_player.NetId == playerId)
		{
			RefreshHoverDisplay();
		}
	}

	private void RefreshHoverDisplay()
	{
		if (_isInPlayerChoice)
		{
			return;
		}
		_tween?.Kill();
		HideThinkyDots();
		AbstractModel abstractModel = RunManager.Instance.HoveredModelTracker.GetHoveredModel(_player.NetId);
		if (NCombatUi.IsDebugHideMpIntents)
		{
			abstractModel = null;
		}
		if (_displayedModel == abstractModel)
		{
			return;
		}
		base.Modulate = StsColors.halfTransparentWhite;
		_cardIntent.Visible = false;
		_relicIntent.Visible = false;
		_potionIntent.Visible = false;
		_powerIntent.Visible = false;
		_hitbox.Visible = abstractModel != null;
		if (abstractModel != null)
		{
			if (!(abstractModel is CardModel card))
			{
				if (!(abstractModel is PotionModel model))
				{
					if (!(abstractModel is RelicModel model2))
					{
						if (!(abstractModel is PowerModel model3))
						{
							throw new InvalidOperationException($"Player {_player.NetId} hovering unsupported model {abstractModel}");
						}
						_powerIntent.Visible = true;
						_powerIntent.Model = model3;
						_hitbox.Position = _powerIntent.Position;
						_hitbox.Size = _powerIntent.Size;
					}
					else
					{
						_relicIntent.Visible = true;
						_relicIntent.Model = model2;
						_hitbox.Position = _relicIntent.Position;
						_hitbox.Size = _relicIntent.Size;
					}
				}
				else
				{
					_potionIntent.Visible = true;
					_potionIntent.Model = model;
					_hitbox.Position = _potionIntent.Position;
					_hitbox.Size = _potionIntent.Size;
				}
			}
			else
			{
				_cardIntent.Visible = true;
				_cardIntent.Card = card;
				_hitbox.Position = _cardIntent.Position;
				_hitbox.Size = _cardIntent.Size;
			}
		}
		RefreshHoverTips();
		_displayedModel = abstractModel;
	}

	private void OnPeerInputStateChanged(ulong playerId)
	{
		if (playerId == _player.NetId)
		{
			bool isTargeting = RunManager.Instance.InputSynchronizer.GetIsTargeting(_player.NetId);
			if (isTargeting && !_targetingIndicator.Visible)
			{
				_targetingIndicator.StartDrawingFrom(Vector2.Zero);
				base.ProcessMode = ProcessModeEnum.Inherit;
			}
			else if (!isTargeting && _targetingIndicator.Visible)
			{
				_targetingIndicator.StopDrawing();
				base.ProcessMode = ProcessModeEnum.Disabled;
			}
		}
	}

	private void OnPeerInputStateRemoved(ulong playerId)
	{
		if (playerId == _player.NetId && _targetingIndicator.Visible)
		{
			_targetingIndicator.StopDrawing();
			base.ProcessMode = ProcessModeEnum.Disabled;
		}
	}

	public override void _Process(double delta)
	{
		Vector2 cursorPosition = NGame.Instance.RemoteCursorContainer.GetCursorPosition(_player.NetId);
		_targetingIndicator.UpdateDrawingTo(cursorPosition - _targetingIndicator.GlobalPosition);
	}

	private void OnActionEnqueued(GameAction action)
	{
		if (action.OwnerId == _player.NetId)
		{
			action.BeforeExecuted += BeforeActionExecuted;
		}
	}

	private void BeforeActionExecuted(GameAction action)
	{
		action.BeforeExecuted -= BeforeActionExecuted;
		action.BeforePausedForPlayerChoice += BeforeActionPausedForPlayerChoice;
		action.BeforeReadyToResumeAfterPlayerChoice += BeforeActionReadyToResumeAfterPlayerChoice;
		action.AfterFinished += UnsubscribeFromAction;
		action.BeforeCancelled += UnsubscribeFromAction;
	}

	private void UnsubscribeFromAction(GameAction action)
	{
		action.BeforePausedForPlayerChoice -= BeforeActionPausedForPlayerChoice;
		action.BeforeReadyToResumeAfterPlayerChoice -= BeforeActionReadyToResumeAfterPlayerChoice;
		action.AfterFinished -= UnsubscribeFromAction;
		action.BeforeCancelled -= UnsubscribeFromAction;
	}

	private void BeforeActionPausedForPlayerChoice(GameAction action)
	{
		AbstractModel abstractModel = null;
		if (action is PlayCardAction playCardAction)
		{
			abstractModel = playCardAction.PlayerChoiceContext?.LastInvolvedModel;
		}
		else if (action is UsePotionAction usePotionAction)
		{
			abstractModel = usePotionAction.PlayerChoiceContext?.LastInvolvedModel;
		}
		else if (action is GenericHookGameAction genericHookGameAction)
		{
			abstractModel = genericHookGameAction.ChoiceContext?.LastInvolvedModel;
		}
		if (abstractModel == null)
		{
			return;
		}
		_isInPlayerChoice = true;
		_cardIntent.Visible = false;
		_relicIntent.Visible = false;
		_potionIntent.Visible = false;
		_powerIntent.Visible = false;
		_hitbox.Visible = false;
		_cardInPlayAwaitingPlayerChoice = null;
		if (abstractModel is CardModel card)
		{
			NCard nCard = NCard.FindOnTable(card);
			_cardThinkyDots.Visible = true;
			_cardThinkyDots.ProcessMode = ProcessModeEnum.Always;
			if (nCard != null)
			{
				nCard.PlayPileTween?.FastForwardToCompletion();
				Tween tween = nCard.CreateTween();
				tween.Parallel().TweenProperty(nCard, "position", _cardIntent.GlobalPosition + _cardIntent.Size / 2f, (SaveManager.Instance.PrefsSave.FastMode == FastModeType.Fast) ? 0.2f : 0.3f);
				tween.Parallel().TweenProperty(nCard, "scale", Vector2.One * 0.25f, (SaveManager.Instance.PrefsSave.FastMode == FastModeType.Fast) ? 0.2f : 0.3f);
				_cardInPlayAwaitingPlayerChoice = nCard;
				_cardThinkyDots.Reparent(nCard.GetParent());
				_hitbox.Reparent(nCard.GetParent());
			}
			else
			{
				_cardIntent.Card = card;
				_cardIntent.Visible = true;
			}
			_hitbox.Visible = true;
			_hitbox.GlobalPosition = _cardIntent.GlobalPosition;
			_hitbox.Size = _cardIntent.Size;
		}
		else if (abstractModel is RelicModel model)
		{
			_relicIntent.Model = model;
			_relicIntent.Visible = true;
			_relicThinkyDots.Visible = true;
			_relicThinkyDots.ProcessMode = ProcessModeEnum.Always;
			_hitbox.Visible = true;
			_hitbox.Position = _relicIntent.Position;
			_hitbox.Size = _relicIntent.Size;
		}
		else if (abstractModel is PotionModel model2)
		{
			_potionIntent.Model = model2;
			_potionIntent.Visible = true;
			_potionThinkyDots.Visible = true;
			_potionThinkyDots.ProcessMode = ProcessModeEnum.Always;
			_hitbox.Visible = true;
			_hitbox.Position = _potionIntent.Position;
			_hitbox.Size = _potionIntent.Size;
		}
		else if (abstractModel is PowerModel model3)
		{
			_powerIntent.Model = model3;
			_powerIntent.Visible = true;
			_powerThinkyDots.Visible = true;
			_powerThinkyDots.ProcessMode = ProcessModeEnum.Always;
			_hitbox.Visible = true;
			_hitbox.Position = _powerIntent.Position;
			_hitbox.Size = _powerIntent.Size;
		}
		RefreshHoverTips();
		base.Modulate = StsColors.transparentWhite;
		_tween?.Kill();
		_tween = GetTree().CreateTween();
		_tween.TweenProperty(this, "modulate", Colors.White, 0.25);
	}

	private void BeforeActionReadyToResumeAfterPlayerChoice(GameAction action)
	{
		_tween?.Kill();
		_tween = GetTree().CreateTween();
		_tween.TweenProperty(this, "modulate", StsColors.transparentWhite, 0.15000000596046448);
		_tween.TweenCallback(Callable.From(HideThinkyDots));
		_isInPlayerChoice = false;
		if (_cardInPlayAwaitingPlayerChoice != null)
		{
			_cardThinkyDots.Reparent(_cardIntent);
			_hitbox.Reparent(this);
			NCardPlayQueue.Instance.ReAddCardAfterPlayerChoice(_cardInPlayAwaitingPlayerChoice, action);
			_cardInPlayAwaitingPlayerChoice = null;
			RefreshHoverTips();
		}
	}

	private void HideThinkyDots()
	{
		_cardThinkyDots.Visible = false;
		_relicThinkyDots.Visible = false;
		_potionThinkyDots.Visible = false;
		_powerThinkyDots.Visible = false;
		_cardThinkyDots.ProcessMode = ProcessModeEnum.Disabled;
		_relicThinkyDots.ProcessMode = ProcessModeEnum.Disabled;
		_potionThinkyDots.ProcessMode = ProcessModeEnum.Disabled;
		_powerThinkyDots.ProcessMode = ProcessModeEnum.Disabled;
	}

	private void RefreshHoverTips()
	{
		if (LocalContext.IsMe(_player))
		{
			return;
		}
		if (NCombatUi.IsDebugHideTargetingUi)
		{
			_shouldShowHoverTip = false;
		}
		else if (!_hitbox.Visible)
		{
			_shouldShowHoverTip = false;
		}
		NHoverTipSet.Remove(this);
		if (_shouldShowHoverTip)
		{
			List<IHoverTip> list = new List<IHoverTip>();
			if (_cardInPlayAwaitingPlayerChoice != null)
			{
				list.Add(HoverTipFactory.FromCard(_cardInPlayAwaitingPlayerChoice.Model));
				list.AddRange(_cardInPlayAwaitingPlayerChoice.Model.HoverTips);
			}
			else if (_cardIntent.Visible)
			{
				list.Add(HoverTipFactory.FromCard(_cardIntent.Card));
				list.AddRange(_cardIntent.Card.HoverTips);
			}
			else if (_relicIntent.Visible)
			{
				list.AddRange(_relicIntent.Model.HoverTips);
			}
			else if (_potionIntent.Visible)
			{
				list.Add(HoverTipFactory.FromPotion(_potionIntent.Model));
				list.AddRange(_potionIntent.Model.HoverTips);
			}
			if (list.Count > 0)
			{
				NHoverTipSet.CreateAndShow(this, list, HoverTipAlignment.Right);
			}
		}
	}
}
