using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NCardFlyPowerVfx : Node2D
{
	private const float _speed = 3000f;

	private const float _scaleOutProportion = 0.9f;

	private const float _initialRotationSpeed = (float)Math.PI;

	private const float _maxRotationSpeed = (float)Math.PI * 50f;

	private NCreature _cardOwnerNode;

	private NCardTrailVfx? _vfx;

	private Path2D _swooshPath;

	private readonly CancellationTokenSource _cancelToken = new CancellationTokenSource();

	private static readonly string _scenePath = SceneHelper.GetScenePath("vfx/vfx_card_power_fly");

	public NCard CardNode { get; private set; }

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public static NCardFlyPowerVfx? Create(NCard card)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCardFlyPowerVfx nCardFlyPowerVfx = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NCardFlyPowerVfx>(PackedScene.GenEditState.Disabled);
		nCardFlyPowerVfx.CardNode = card;
		return nCardFlyPowerVfx;
	}

	public override void _Ready()
	{
		base.GlobalPosition = CardNode.GlobalPosition;
		Player owner = CardNode.Model.Owner;
		_cardOwnerNode = NCombatRoom.Instance.GetCreatureNode(owner.Creature);
		_vfx = NCardTrailVfx.Create(CardNode, owner.Character.TrailPath);
		if (_vfx != null)
		{
			this.AddChildSafely(_vfx);
		}
		Vector2 vfxSpawnPosition = _cardOwnerNode.VfxSpawnPosition;
		Vector2 position = vfxSpawnPosition - base.GlobalPosition;
		_swooshPath = GetNode<Path2D>("SwooshPath");
		_swooshPath.Curve.SetPointPosition(_swooshPath.Curve.PointCount - 1, position);
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		_cancelToken.Cancel();
		_cancelToken.Dispose();
	}

	public float GetDuration()
	{
		return GetDurationInternal() + 0.05f;
	}

	private float GetDurationInternal()
	{
		return _swooshPath.Curve.GetBakedLength() / 3000f;
	}

	public async Task PlayAnim()
	{
		CreateTween().TweenProperty(CardNode, "scale", Vector2.One * 0.1f, 0.30000001192092896);
		float length = _swooshPath.Curve.GetBakedLength();
		double timeAccumulator = 0.0;
		float duration = GetDurationInternal();
		while (timeAccumulator < (double)duration)
		{
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
			if (_cancelToken.IsCancellationRequested)
			{
				break;
			}
			double processDeltaTime = GetProcessDeltaTime();
			timeAccumulator += processDeltaTime;
			float num = (float)(timeAccumulator / (double)duration);
			float num2 = Ease.QuadIn(num);
			Transform2D transform2D = _swooshPath.Curve.SampleBakedWithRotation(num2 * length);
			CardNode.GlobalPosition = base.GlobalPosition + transform2D.Origin;
			float s = transform2D.Rotation - CardNode.Rotation;
			float num3 = Mathf.Lerp((float)Math.PI, (float)Math.PI * 50f, num);
			CardNode.Rotation += (float)Mathf.Sign(s) * Mathf.Min(Mathf.Abs(s), (float)((double)num3 * processDeltaTime));
			if (num >= 0.9f)
			{
				CreateTween().TweenProperty(CardNode, "scale", Vector2.Zero, (double)duration - timeAccumulator);
			}
		}
		NGame.Instance.ScreenShake(ShakeStrength.Medium, ShakeDuration.Short);
		if (_vfx != null)
		{
			await _vfx.FadeOut();
		}
		CardNode.QueueFreeSafely();
		this.QueueFreeSafely();
	}
}
