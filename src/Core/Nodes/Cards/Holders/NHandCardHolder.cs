using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Extensions;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Cards.Holders;

public partial class NHandCardHolder : NCardHolder
{
	[Signal]
	public delegate void HolderFocusedEventHandler(NHandCardHolder cardHolder);

	[Signal]
	public delegate void HolderUnfocusedEventHandler(NHandCardHolder cardHolder);

	[Signal]
	public delegate void HolderMouseClickedEventHandler(NCardHolder cardHolder);

	private Control _flash;

	private Tween? _flashTween;

	private MegaLabel _handIndexLabel;

	private const float _rotateSpeed = 10f;

	private const float _angleSnapThreshold = 0.1f;

	private const float _scaleSpeed = 8f;

	private const float _scaleSnapThreshold = 0.002f;

	private const float _moveSpeed = 7f;

	private const float _positionSnapThreshold = 1f;

	private const float _reenableHitboxThreshold = 200f;

	private Vector2 _targetPosition;

	private float _targetAngle;

	private Vector2 _targetScale;

	private CancellationTokenSource? _angleCancelToken;

	private CancellationTokenSource? _positionCancelToken;

	private CancellationTokenSource? _scaleCancelToken;

	private NPlayerHand _hand;

	public bool InSelectMode { get; set; }

	public Vector2 TargetPosition => _targetPosition;

	public float TargetAngle => _targetAngle;

	private static string ScenePath => SceneHelper.GetScenePath("cards/holders/hand_card_holder");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	private bool ShouldGlowGold
	{
		get
		{
			CardModel cardModel = base.CardNode?.Model;
			if (cardModel == null)
			{
				return false;
			}
			if (_hand.SelectModeGoldGlowOverride != null)
			{
				return _hand.SelectModeGoldGlowOverride(cardModel);
			}
			if (CombatManager.Instance.IsPlayPhase && cardModel.CanPlay())
			{
				return cardModel.ShouldGlowGold;
			}
			return false;
		}
	}

	private bool ShouldGlowRed
	{
		get
		{
			CardModel cardModel = base.CardNode?.Model;
			if (cardModel == null)
			{
				return false;
			}
			if (CombatManager.Instance.IsPlayPhase)
			{
				return cardModel.ShouldGlowRed;
			}
			return false;
		}
	}

	public static NHandCardHolder Create(NCard card, NPlayerHand hand)
	{
		NHandCardHolder nHandCardHolder = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NHandCardHolder>(PackedScene.GenEditState.Disabled);
		nHandCardHolder.Name = $"{nHandCardHolder.GetType().Name}-{card.Model.Id}";
		nHandCardHolder.SetCard(card);
		nHandCardHolder._hand = hand;
		return nHandCardHolder;
	}

	public override void _Ready()
	{
		ConnectSignals();
		_flash = GetNode<Control>("Flash");
		_flash.Modulate = new Color(_flash.Modulate.R, _flash.Modulate.G, _flash.Modulate.B, 0f);
		_handIndexLabel = GetNode<MegaLabel>("%HandIndex");
		UpdateCard();
		base.Hitbox.SetEnabled(enabled: false);
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		UnsubscribeFromEvents(base.CardNode?.Model);
		StopAnimations();
	}

	public override void Clear()
	{
		UnsubscribeFromEvents(base.CardNode?.Model);
		base.Clear();
		StopAnimations();
	}

	protected override void OnFocus()
	{
		EmitSignal(SignalName.HolderFocused, this);
		base.OnFocus();
	}

	protected override void OnUnfocus()
	{
		EmitSignal(SignalName.HolderUnfocused, this);
		base.OnUnfocus();
	}

	protected override void OnMousePressed(InputEvent inputEvent)
	{
		if (inputEvent is InputEventMouseButton inputEventMouseButton && inputEventMouseButton.ButtonIndex == MouseButton.Left && _isClickable)
		{
			SfxCmd.Play("event:/sfx/ui/clicks/ui_click");
			EmitSignal(SignalName.HolderMouseClicked, this);
		}
	}

	protected override void OnMouseReleased(InputEvent inputEvent)
	{
	}

	protected override void DoCardHoverEffects(bool isHovered)
	{
		base.ZIndex = (isHovered ? 1 : 0);
		if (isHovered)
		{
			CreateHoverTips();
		}
		else
		{
			ClearHoverTips();
		}
	}

	public void SetIndexLabel(int i)
	{
		_handIndexLabel.SetTextAutoSize(i.ToString());
		_handIndexLabel.Visible = i > 0 && SaveManager.Instance.PrefsSave.ShowCardIndices;
	}

	public void SetTargetAngle(float angle)
	{
		_targetAngle = angle;
		_angleCancelToken?.Cancel();
		_angleCancelToken = new CancellationTokenSource();
		TaskHelper.RunSafely(AnimAngle(_angleCancelToken));
	}

	public void SetTargetPosition(Vector2 position)
	{
		_targetPosition = position;
		_positionCancelToken?.Cancel();
		_positionCancelToken = new CancellationTokenSource();
		TaskHelper.RunSafely(AnimPosition(_positionCancelToken));
	}

	public void SetTargetScale(Vector2 scale)
	{
		_targetScale = scale;
		_scaleCancelToken?.Cancel();
		_scaleCancelToken = new CancellationTokenSource();
		TaskHelper.RunSafely(AnimScale(_scaleCancelToken));
	}

	public void SetAngleInstantly(float setAngle)
	{
		_angleCancelToken?.Cancel();
		base.RotationDegrees = setAngle;
	}

	public void SetScaleInstantly(Vector2 setScale)
	{
		_scaleCancelToken?.Cancel();
		base.Scale = setScale;
	}

	private void StopAnimations()
	{
		_angleCancelToken?.Cancel();
		_positionCancelToken?.Cancel();
		_scaleCancelToken?.Cancel();
	}

	private async Task AnimAngle(CancellationTokenSource cancelToken)
	{
		while (!cancelToken.IsCancellationRequested)
		{
			base.RotationDegrees = Mathf.Lerp(base.RotationDegrees, _targetAngle, (float)GetProcessDeltaTime() * 10f);
			if (Mathf.Abs(base.RotationDegrees - _targetAngle) < 0.1f)
			{
				base.RotationDegrees = _targetAngle;
				break;
			}
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		}
	}

	private async Task AnimScale(CancellationTokenSource cancelToken)
	{
		while (!cancelToken.IsCancellationRequested)
		{
			base.Scale = base.Scale.Lerp(_targetScale, (float)GetProcessDeltaTime() * 8f);
			if (Mathf.Abs(_targetScale.X - base.Scale.X) < 0.002f)
			{
				base.Scale = _targetScale;
				break;
			}
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		}
	}

	private async Task AnimPosition(CancellationTokenSource cancelToken)
	{
		while (!cancelToken.IsCancellationRequested)
		{
			base.Position = base.Position.Lerp(_targetPosition, (float)GetProcessDeltaTime() * 7f);
			float num = Mathf.Abs(base.Position.X - _targetPosition.X);
			if (!base.Hitbox.IsEnabled && num < 200f)
			{
				base.Hitbox.SetEnabled(enabled: true);
			}
			if (base.Position.DistanceSquaredTo(_targetPosition) < 1f)
			{
				base.Position = _targetPosition;
				return;
			}
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		}
		if (!base.Hitbox.IsEnabled && base.Position.DistanceSquaredTo(_targetPosition) < 200f)
		{
			base.Hitbox.SetEnabled(enabled: true);
		}
	}

	protected override void SetCard(NCard node)
	{
		if (base.CardNode != null)
		{
			base.CardNode.ModelChanged -= OnModelChanged;
		}
		UnsubscribeFromEvents(base.CardNode?.Model);
		base.SetCard(node);
		UpdateCard();
		SubscribeToEvents(base.CardNode?.Model);
		if (base.CardNode != null)
		{
			base.CardNode.ModelChanged += OnModelChanged;
		}
		if (node.Scale != Vector2.One)
		{
			node.CreateTween().TweenProperty(node, "scale", Vector2.One, 0.25);
		}
	}

	public void UpdateCard()
	{
		if (!IsNodeReady() || base.CardNode == null)
		{
			return;
		}
		base.CardNode.UpdateVisuals(PileType.Hand, CardPreviewMode.Normal);
		if (!CombatManager.Instance.IsInProgress)
		{
			return;
		}
		if (base.CardNode.Model.CanPlay() || ShouldGlowRed || ShouldGlowGold)
		{
			base.CardNode.CardHighlight.AnimShow();
			base.CardNode.CardHighlight.Modulate = NCardHighlight.playableColor;
			if (ShouldGlowRed)
			{
				base.CardNode.CardHighlight.Modulate = NCardHighlight.red;
			}
			else if (ShouldGlowGold)
			{
				base.CardNode.CardHighlight.Modulate = NCardHighlight.gold;
			}
		}
		else
		{
			base.CardNode.CardHighlight.AnimHide();
		}
	}

	public void BeginDrag()
	{
		SetAngleInstantly(0f);
		SetScaleInstantly(HoverScale);
	}

	public void CancelDrag()
	{
		base.ZIndex = 0;
		SetAngleInstantly(0f);
		SetScaleInstantly(Vector2.One);
	}

	public void SetDefaultTargets()
	{
		base.ZIndex = 0;
		IReadOnlyList<NHandCardHolder> activeHolders = _hand.ActiveHolders;
		int num = activeHolders.IndexOf(this);
		int count = activeHolders.Count;
		if (num >= 0)
		{
			SetTargetPosition(HandPosHelper.GetPosition(count, num));
			SetTargetAngle(HandPosHelper.GetAngle(count, num));
			SetTargetScale(HandPosHelper.GetScale(count));
		}
	}

	public void Flash()
	{
		if (GodotObject.IsInstanceValid(_flash))
		{
			_flash.Scale = Vector2.One;
			_flash.Modulate = NCardHighlight.playableColor;
			if (ShouldGlowGold)
			{
				_flash.Modulate = NCardHighlight.gold;
			}
			else if (ShouldGlowRed)
			{
				_flash.Modulate = NCardHighlight.red;
			}
			_flashTween?.Kill();
			_flashTween = CreateTween();
			_flashTween.TweenProperty(_flash, "modulate:a", 0.6, 0.15);
			_flashTween.TweenProperty(_flash, "modulate:a", 0, 0.3);
		}
	}

	private void SubscribeToEvents(CardModel? card)
	{
		if (card != null && IsInsideTree())
		{
			card.Upgraded += Flash;
			card.KeywordsChanged += Flash;
			card.ReplayCountChanged += Flash;
			card.AfflictionChanged += Flash;
			card.EnergyCostChanged += Flash;
			card.StarCostChanged += Flash;
		}
	}

	private void UnsubscribeFromEvents(CardModel? card)
	{
		if (card != null)
		{
			card.Upgraded -= Flash;
			card.KeywordsChanged -= Flash;
			card.ReplayCountChanged -= Flash;
			card.AfflictionChanged -= Flash;
			card.EnergyCostChanged -= Flash;
			card.StarCostChanged -= Flash;
		}
	}

	private void OnModelChanged(CardModel? oldModel)
	{
		UnsubscribeFromEvents(oldModel);
		SubscribeToEvents(base.CardNode?.Model);
	}
}
