using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NCardTransformVfx : Node2D
{
	private Tween? _tween;

	private CardModel _startCard;

	private CardModel _endCard;

	private IEnumerable<RelicModel>? _relicsToFlash;

	private static string ScenePath => SceneHelper.GetScenePath("vfx/vfx_card_transform");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	public static NCardTransformVfx? Create(CardModel startCard, CardModel endCard, IEnumerable<RelicModel>? relicsToFlash)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCardTransformVfx nCardTransformVfx = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NCardTransformVfx>(PackedScene.GenEditState.Disabled);
		nCardTransformVfx._startCard = startCard;
		nCardTransformVfx._endCard = endCard;
		nCardTransformVfx._relicsToFlash = relicsToFlash;
		return nCardTransformVfx;
	}

	public override void _Ready()
	{
		TaskHelper.RunSafely(PlayAnimation());
	}

	private async Task<bool> WaitAndInterruptIfNecessary(float seconds, NCard cardNode)
	{
		for (float timer = 0f; timer <= seconds; timer += (float)GetProcessDeltaTime())
		{
			if (!cardNode.IsInsideTree() || _endCard.Pile == null)
			{
				return false;
			}
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		}
		return true;
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
	}

	private async Task PlayAnimation()
	{
		SfxCmd.Play("event:/sfx/ui/cards/card_transform");
		Material textureMat = GetNode<Sprite2D>("%RenderTexture").Material;
		NCard cardNode = NCard.Create(_startCard);
		SubViewport node = GetNode<SubViewport>("SubViewport");
		node.AddChildSafely(cardNode);
		cardNode.UpdateVisuals(PileType.None, CardPreviewMode.Normal);
		cardNode.Position = new Vector2((float)node.Size.X * 0.5f, (float)node.Size.Y * 0.5f);
		_tween = CreateTween();
		_tween.TweenProperty(cardNode, "scale", Vector2.One * 1f, 0.25).From(Vector2.Zero).SetEase(Tween.EaseType.Out)
			.SetTrans(Tween.TransitionType.Cubic);
		if (!(await WaitAndInterruptIfNecessary(0.75f, cardNode)))
		{
			this.QueueFreeSafely();
			return;
		}
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(textureMat, "shader_parameter/brightness", 1f, 0.5);
		_tween.TweenProperty(textureMat, "shader_parameter/boing:x", 2f, 0.4000000059604645);
		if (!(await WaitAndInterruptIfNecessary(0.5f, cardNode)))
		{
			this.QueueFreeSafely();
			return;
		}
		cardNode.Model = _endCard;
		cardNode.UpdateVisuals(PileType.None, CardPreviewMode.Normal);
		GetNode<CpuParticles2D>("%Particle").Emitting = true;
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(textureMat, "shader_parameter/brightness", 0f, 0.20000000298023224);
		_tween.TweenProperty(textureMat, "shader_parameter/boing:x", -0.75f, 0.15000000596046448).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Quad);
		await ToSignal(_tween, Tween.SignalName.Finished);
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(textureMat, "shader_parameter/boing:x", 0.3f, 0.20000000298023224).SetEase(Tween.EaseType.InOut).SetTrans(Tween.TransitionType.Quad);
		await ToSignal(_tween, Tween.SignalName.Finished);
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(textureMat, "shader_parameter/boing:x", -0.2f, 0.25).SetEase(Tween.EaseType.InOut).SetTrans(Tween.TransitionType.Quad);
		await ToSignal(_tween, Tween.SignalName.Finished);
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(textureMat, "shader_parameter/boing:x", 0, 0.30000001192092896).SetEase(Tween.EaseType.InOut).SetTrans(Tween.TransitionType.Back);
		if (!(await WaitAndInterruptIfNecessary(0.3f, cardNode)))
		{
			this.QueueFreeSafely();
			return;
		}
		if (_relicsToFlash != null)
		{
			foreach (RelicModel item in _relicsToFlash)
			{
				item.Flash();
				cardNode.FlashRelicOnCard(item);
			}
		}
		if (!(await WaitAndInterruptIfNecessary(0.5f, cardNode)))
		{
			this.QueueFreeSafely();
			return;
		}
		Vector2 targetPosition = _endCard.Pile.Type.GetTargetPosition(cardNode);
		cardNode.Reparent(this);
		cardNode.Position = Vector2.Zero;
		NCardFlyVfx nCardFlyVfx = NCardFlyVfx.Create(cardNode, targetPosition, isAddingToPile: false, _endCard.Owner.Character.TrailPath);
		CardPile? pile = _endCard.Pile;
		((pile == null || pile.Type != PileType.Deck) ? NCombatRoom.Instance?.CombatVfxContainer : NRun.Instance?.GlobalUi.TopBar.TrailContainer)?.AddChildSafely(nCardFlyVfx);
		if (nCardFlyVfx?.SwooshAwayCompletion != null)
		{
			await nCardFlyVfx.SwooshAwayCompletion.Task;
		}
		this.QueueFreeSafely();
	}

	public static async Task PlayAnimOnCardInHand(NCard cardNode, CardModel endCard)
	{
		if (!TestMode.IsOn)
		{
			SfxCmd.Play("event:/sfx/ui/cards/card_transform");
			Tween tween = cardNode.CreateTween();
			tween.TweenProperty(cardNode, "scale", Vector2.One * 1.5f, 0.25).From(Vector2.One).SetEase(Tween.EaseType.Out)
				.SetTrans(Tween.TransitionType.Cubic);
			await cardNode.ToSignal(tween, Tween.SignalName.Finished);
			NPlayerHand.Instance?.TryCancelCardPlay(cardNode.Model);
			cardNode.Model = endCard;
			cardNode.UpdateVisuals(endCard.Pile.Type, CardPreviewMode.Normal);
			Tween tween2 = cardNode.CreateTween();
			tween2.TweenProperty(cardNode, "scale", Vector2.One, 0.25).From(Vector2.One * 1.5f).SetEase(Tween.EaseType.In)
				.SetTrans(Tween.TransitionType.Cubic);
			if (NCombatRoom.Instance?.Ui.Hand.GetCardHolder(endCard) is NHandCardHolder nHandCardHolder)
			{
				nHandCardHolder.UpdateCard();
			}
			await cardNode.ToSignal(tween2, Tween.SignalName.Finished);
		}
	}
}
