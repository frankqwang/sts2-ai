using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NCardFlyShuffleVfx : Control
{
	private NCardTrailVfx? _vfx;

	private Tween? _fadeOutTween;

	private bool _vfxFading;

	private Vector2 _startPos;

	private Vector2 _endPos;

	private float _controlPointOffset;

	private float _duration;

	private float _speed;

	private float _accel;

	private float _arcDir;

	private string _trailPath;

	private CardPile _targetPile;

	private readonly CancellationTokenSource _cancelToken = new CancellationTokenSource();

	private static readonly string _scenePath = SceneHelper.GetScenePath("vfx/vfx_card_shuffle_fly");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public static NCardFlyShuffleVfx? Create(CardPile startPile, CardPile targetPile, string trailPath)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCardFlyShuffleVfx nCardFlyShuffleVfx = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NCardFlyShuffleVfx>(PackedScene.GenEditState.Disabled);
		nCardFlyShuffleVfx._startPos = startPile.Type.GetTargetPosition(null);
		nCardFlyShuffleVfx._endPos = targetPile.Type.GetTargetPosition(null);
		nCardFlyShuffleVfx._trailPath = trailPath;
		nCardFlyShuffleVfx._targetPile = targetPile;
		return nCardFlyShuffleVfx;
	}

	public override void _Ready()
	{
		_controlPointOffset = Rng.Chaotic.NextFloat(-300f, 400f);
		_speed = Rng.Chaotic.NextFloat(1.1f, 1.25f);
		_accel = Rng.Chaotic.NextFloat(2f, 2.5f);
		_arcDir = ((_endPos.Y < 540f) ? (-500f) : (500f + _controlPointOffset));
		_duration = Rng.Chaotic.NextFloat(1f, 1.75f);
		_vfx = NCardTrailVfx.Create(this, _trailPath);
		if (_vfx != null)
		{
			NCombatRoom.Instance.CombatVfxContainer.AddChildSafely(_vfx);
		}
		Node parent = GetParent();
		parent.MoveChild(this, parent.GetChildCount() - 1);
		TaskHelper.RunSafely(PlayAnim());
	}

	private async Task PlayAnim()
	{
		float time = 0f;
		while (time / _duration <= 1f)
		{
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
			if (_cancelToken.IsCancellationRequested)
			{
				return;
			}
			float num = (float)GetProcessDeltaTime();
			time += _speed * num;
			_speed += _accel * num;
			Vector2 c = _startPos + (_endPos - _startPos) * 0.5f;
			c.Y -= _arcDir;
			base.GlobalPosition = MathHelper.BezierCurve(_startPos, _endPos, c, time / _duration);
			Vector2 vector = MathHelper.BezierCurve(_startPos, _endPos, c, (time + 0.05f) / _duration);
			base.Rotation = (vector - base.GlobalPosition).Angle() + (float)Math.PI / 2f;
		}
		base.GlobalPosition = _endPos;
		_targetPile.InvokeCardAddFinished();
		time = 0f;
		while (time / _duration <= 1f)
		{
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
			if (_cancelToken.IsCancellationRequested)
			{
				return;
			}
			float num2 = (float)GetProcessDeltaTime();
			time += _speed * num2;
			if (time / _duration > 0.25f && !_vfxFading)
			{
				if (_vfx != null)
				{
					TaskHelper.RunSafely(_vfx.FadeOut());
				}
				_vfxFading = true;
			}
			base.Scale = Vector2.One * Mathf.Max(Mathf.Lerp(0.1f, -0.1f, time / _duration), 0f);
		}
		_fadeOutTween = CreateTween();
		_fadeOutTween.TweenProperty(this, "modulate:a", 0f, 0.800000011920929);
		await Task.Delay(800);
		this.QueueFreeSafely();
	}

	public override void _ExitTree()
	{
		_fadeOutTween?.Kill();
		_cancelToken.Cancel();
		_cancelToken.Dispose();
	}
}
