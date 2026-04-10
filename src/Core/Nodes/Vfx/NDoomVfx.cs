using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NDoomVfx : Node2D
{
	private Tween? _tween;

	private NDoomSubEmitterVfx _back;

	private NDoomSubEmitterVfx _front;

	private NCreatureVisuals _creatureVisuals;

	private Vector2 _position;

	private Vector2 _size;

	private bool _shouldDie;

	private CancellationToken _cancelToken;

	private const float _doomVfxSize = 260f;

	private CancellationTokenSource VfxCancellationToken { get; } = new CancellationTokenSource();

	private static string ScenePath => SceneHelper.GetScenePath("vfx/vfx_doom");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	public Task? VfxTask { get; private set; }

	public static NDoomVfx? Create(NCreatureVisuals creatureVisuals, Vector2 position, Vector2 size, bool shouldDie)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NDoomVfx nDoomVfx = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NDoomVfx>(PackedScene.GenEditState.Disabled);
		nDoomVfx._creatureVisuals = creatureVisuals;
		nDoomVfx._position = position;
		nDoomVfx._size = size;
		nDoomVfx._shouldDie = shouldDie;
		return nDoomVfx;
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		VfxCancellationToken.Cancel();
		_tween?.Kill();
	}

	public override void _Ready()
	{
		_back = GetNode<NDoomSubEmitterVfx>("DoomVfxBack");
		_front = GetNode<NDoomSubEmitterVfx>("DoomVfxFront");
		_cancelToken = VfxCancellationToken.Token;
		VfxTask = TaskHelper.RunSafely(PlayVfx(_creatureVisuals, _position, _size, _shouldDie));
	}

	private async Task PlayVfx(NCreatureVisuals creatureVisuals, Vector2 position, Vector2 size, bool shouldDie)
	{
		if (!_cancelToken.IsCancellationRequested)
		{
			SfxCmd.Play("event:/sfx/characters/necrobinder/necrobinder_doom_kill");
			base.GlobalPosition = position + new Vector2(size.X * 0.5f, size.Y) * NCombatRoom.Instance.SceneContainer.Scale;
			base.Scale = NCombatRoom.Instance.SceneContainer.Scale;
			SubViewport node = GetNode<SubViewport>("Viewport");
			Vector2 vector = size;
			vector.X *= 1.5f;
			vector.Y *= 1.5f;
			node.Size = (Vector2I)vector;
			if (shouldDie)
			{
				Vector2 creatureOffset = new Vector2(vector.X / 2f, node.Size.Y) + creatureVisuals.Body.Position;
				Vector2 originalGlobalScale = creatureVisuals.Body.GlobalScale;
				await Reparent(creatureVisuals.Body, node);
				creatureVisuals.Body.Position = creatureOffset;
				creatureVisuals.Body.Scale = originalGlobalScale;
			}
			if (!_cancelToken.IsCancellationRequested)
			{
				await PlayVfxInternal();
			}
		}
	}

	private async Task PlayVfxInternal()
	{
		_ = 1;
		try
		{
			SubViewport node = GetNode<SubViewport>("Viewport");
			Sprite2D node2 = GetNode<Sprite2D>("%Visual");
			node2.Position += Vector2.Up * node.Size.Y * 0.5f;
			NGame.Instance?.ScreenShake(ShakeStrength.Weak, ShakeDuration.Short, 180f + Rng.Chaotic.NextFloat(-10f, 10f));
			ShowOrHideParticles((float)node.Size.X / 260f, 0.5f);
			_tween = CreateTween();
			_tween.TweenProperty(node2, "position:y", node2.Position.Y + (float)node.Size.Y, 0.75).SetEase(Tween.EaseType.In).SetDelay(0.75)
				.SetTrans(Tween.TransitionType.Expo);
			await ToSignal(_tween, Tween.SignalName.Finished);
			ShowOrHideParticles(0f, 0.25f);
			await Task.Delay(2000, _cancelToken);
		}
		finally
		{
			if (GodotObject.IsInstanceValid(this))
			{
				this.QueueFreeSafely();
			}
		}
	}

	private void ShowOrHideParticles(float widthScale, float tweenTime)
	{
		_back.ShowOrHide(widthScale, tweenTime);
		_front.ShowOrHide(widthScale, tweenTime);
	}

	private async Task Reparent(Node creatureNode, Node newParent)
	{
		Node parent = creatureNode.GetParent();
		bool removeCompleted = false;
		Callable reparent = Callable.From(() => removeCompleted = true);
		creatureNode.Connect(Node.SignalName.TreeExited, reparent);
		parent.RemoveChildSafely(creatureNode);
		while (!removeCompleted)
		{
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		}
		newParent.AddChildSafely(creatureNode);
		creatureNode.Disconnect(Node.SignalName.TreeExited, reparent);
	}
}
