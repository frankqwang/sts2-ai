using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NCardFlyVfx : Node2D
{
	private NCard _card;

	private string _trailPath;

	private NCardTrailVfx? _vfx;

	private Tween? _fadeOutTween;

	private bool _vfxFading;

	private bool _isAddingToPile;

	private Vector2 _startPos;

	private Vector2 _endPos;

	private float _controlPointOffset;

	private float _duration;

	private float _speed;

	private float _accel;

	private float _arcDir;

	private readonly CancellationTokenSource _cancelToken = new CancellationTokenSource();

	private static readonly string _scenePath = SceneHelper.GetScenePath("vfx/vfx_card_fly");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public TaskCompletionSource? SwooshAwayCompletion { get; private set; }

	public static NCardFlyVfx? Create(NCard card, Vector2 end, bool isAddingToPile, string trailPath)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCardFlyVfx nCardFlyVfx = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NCardFlyVfx>(PackedScene.GenEditState.Disabled);
		nCardFlyVfx._startPos = card.GlobalPosition;
		nCardFlyVfx._endPos = end;
		nCardFlyVfx._card = card;
		nCardFlyVfx._isAddingToPile = isAddingToPile;
		nCardFlyVfx._trailPath = trailPath;
		return nCardFlyVfx;
	}

	public override void _Ready()
	{
		_vfx = NCardTrailVfx.Create(_card, _trailPath);
		if (_vfx != null)
		{
			GetParent().AddChildSafely(_vfx);
		}
		_controlPointOffset = Rng.Chaotic.NextFloat(100f, 400f);
		_speed = Rng.Chaotic.NextFloat(1.1f, 1.25f);
		_accel = Rng.Chaotic.NextFloat(2f, 2.5f);
		_arcDir = ((_endPos.Y < GetViewportRect().Size.Y * 0.5f) ? (-500f) : (500f + _controlPointOffset));
		_duration = Rng.Chaotic.NextFloat(1f, 1.75f);
		_card.Connect(Node.SignalName.TreeExited, Callable.From(OnCardExitedTree));
		if (NCombatUi.IsDebugHidingPlayContainer)
		{
			_card.Modulate = Colors.Transparent;
			_card.Visible = false;
			base.Visible = false;
		}
		TaskHelper.RunSafely(PlayAnim());
	}

	public override void _ExitTree()
	{
		_fadeOutTween?.Kill();
		_cancelToken.Cancel();
		_cancelToken.Dispose();
	}

	private void OnCardExitedTree()
	{
		try
		{
			_vfx?.QueueFreeSafely();
		}
		catch (ObjectDisposedException)
		{
		}
		SwooshAwayCompletion?.TrySetResult();
		this.QueueFreeSafely();
	}

	private async Task PlayAnim()
	{
		SwooshAwayCompletion = new TaskCompletionSource();
		float time = 0f;
		while (time / _duration <= 1f)
		{
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
			if (_cancelToken.IsCancellationRequested)
			{
				SwooshAwayCompletion?.SetResult();
				return;
			}
			float num = (float)GetProcessDeltaTime();
			time += _speed * num;
			_speed += _accel * num;
			Vector2 c = _startPos + (_endPos - _startPos) * 0.5f;
			c.Y -= _arcDir;
			Vector2 vector = MathHelper.BezierCurve(_startPos, _endPos, c, (time + 0.05f) / _duration);
			_card.GlobalPosition = MathHelper.BezierCurve(_startPos, _endPos, c, time / _duration);
			float num2 = (vector - _card.GlobalPosition).Angle() + (float)Math.PI / 2f;
			Node parent = _card.GetParent();
			if (parent is Control control)
			{
				num2 -= control.Rotation;
			}
			else if (parent is Node2D node2D)
			{
				num2 -= node2D.Rotation;
			}
			_card.Rotation = Mathf.LerpAngle(_card.Rotation, num2, num * 12f);
			_card.Body.Modulate = Colors.White.Lerp(Colors.Black, Mathf.Clamp(time * 3f / _duration, 0f, 1f));
			_card.Body.Scale = Vector2.One * Mathf.Lerp(1f, 0.1f, Mathf.Clamp(time * 3f / _duration, 0f, 1f));
		}
		_card.GlobalPosition = _endPos;
		if (_isAddingToPile)
		{
			_card.Model.Pile?.InvokeCardAddFinished();
		}
		time = 0f;
		while (time / _duration <= 1f)
		{
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
			if (_cancelToken.IsCancellationRequested)
			{
				SwooshAwayCompletion?.SetResult();
				return;
			}
			float num3 = (float)GetProcessDeltaTime();
			time += _speed * num3;
			if (time / _duration > 0.25f && !_vfxFading)
			{
				if (_vfx != null)
				{
					TaskHelper.RunSafely(_vfx.FadeOut());
				}
				_vfxFading = true;
			}
			_card.Body.Scale = Vector2.One * Mathf.Max(Mathf.Lerp(0.1f, -0.15f, time / _duration), 0f);
		}
		SwooshAwayCompletion?.SetResult();
		_card.QueueFreeSafely();
	}
}
