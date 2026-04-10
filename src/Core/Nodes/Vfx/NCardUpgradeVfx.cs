using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NCardUpgradeVfx : Node2D
{
	private CardModel _card;

	private CancellationTokenSource? _cts;

	private static string ScenePath => SceneHelper.GetScenePath("vfx/vfx_card_upgrade");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	public static NCardUpgradeVfx? Create(CardModel card)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCardUpgradeVfx nCardUpgradeVfx = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NCardUpgradeVfx>(PackedScene.GenEditState.Disabled);
		nCardUpgradeVfx._card = card;
		return nCardUpgradeVfx;
	}

	public override void _Ready()
	{
		TaskHelper.RunSafely(PlayAnimation());
	}

	public override void _ExitTree()
	{
		_cts?.Cancel();
		_cts?.Dispose();
	}

	private async Task PlayAnimation()
	{
		_cts = new CancellationTokenSource();
		NCard cardNode = NCard.Create(_card);
		this.AddChildSafely(cardNode);
		MoveChild(cardNode, 0);
		cardNode.UpdateVisuals(PileType.None, CardPreviewMode.Normal);
		GetNode<CpuParticles2D>("%Particle").Emitting = true;
		Tween tween = CreateTween();
		tween.TweenProperty(cardNode, "scale", Vector2.One * 1f, 0.25).From(Vector2.Zero).SetEase(Tween.EaseType.Out)
			.SetTrans(Tween.TransitionType.Cubic);
		await Cmd.Wait(1.75f, _cts.Token);
		Vector2 targetPosition = _card.Pile.Type.GetTargetPosition(cardNode);
		NCardFlyVfx nCardFlyVfx = NCardFlyVfx.Create(cardNode, targetPosition, isAddingToPile: false, _card.Owner.Character.TrailPath);
		((_card.Pile.Type != PileType.Deck) ? NCombatRoom.Instance?.CombatVfxContainer : NRun.Instance?.GlobalUi.TopBar.TrailContainer)?.AddChildSafely(nCardFlyVfx);
		if (nCardFlyVfx?.SwooshAwayCompletion != null)
		{
			await nCardFlyVfx.SwooshAwayCompletion.Task;
		}
		if (!_cts.IsCancellationRequested)
		{
			this.QueueFreeSafely();
		}
	}
}
