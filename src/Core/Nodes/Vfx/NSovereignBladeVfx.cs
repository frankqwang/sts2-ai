using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NSovereignBladeVfx : Node2D
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("vfx/sovereign_blade");

	private Player _owner;

	private Node2D _spineNode;

	private MegaSprite _animController;

	private Node2D _bladeGlow;

	private GpuParticles2D _forgeSparks;

	private GpuParticles2D _spawnFlames;

	private GpuParticles2D _spawnFlamesBack;

	private GpuParticles2D _slashParticles;

	private GpuParticles2D _chargeParticles;

	private GpuParticles2D _spikeParticles;

	private GpuParticles2D _spikeParticles2;

	private GpuParticles2D _spikeCircle;

	private GpuParticles2D _spikeCircle2;

	private TextureRect _hilt;

	private TextureRect _hilt2;

	private TextureRect _detail;

	private Line2D _trail;

	private Path2D _orbitPath;

	private Control _hitbox;

	private NSelectionReticle _selectionReticle;

	private Tween? _attackTween;

	private Tween? _scaleTween;

	private Tween? _sparkDelay;

	private Tween? _glowTween;

	private Vector2 _trailStart;

	private float _bladeSize;

	private const float _orbitSpeed = 60f;

	private Vector2 _targetOrbitPosition;

	private bool _isBehindCharacter;

	private const float _hiltThreshold = 0.3f;

	private const float _detailThreshold = 0.66f;

	private bool _isFocused;

	private NHoverTipSet? _hoverTip;

	private bool _isForging;

	private bool _isAttacking;

	private bool _isKeyPressed;

	private float _testCharge;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public CardModel Card { get; private set; }

	public double OrbitProgress { get; set; }

	public override void _Ready()
	{
		_spineNode = GetNode<Node2D>("SpineSword");
		_animController = new MegaSprite(_spineNode);
		_bladeGlow = GetNode<Node2D>("SpineSword/SwordBone/ScaleContainer/BladeGlow");
		_forgeSparks = GetNode<GpuParticles2D>("SpineSword/SwordBone/ScaleContainer/ForgeSparks");
		_spawnFlames = GetNode<GpuParticles2D>("SpineSword/SwordBone/ScaleContainer/SpawnFlames");
		_spawnFlamesBack = GetNode<GpuParticles2D>("SpineSword/SwordBone/ScaleContainer/SpawnFlamesBack");
		_slashParticles = GetNode<GpuParticles2D>("SpineSword/SlashParticles");
		_chargeParticles = GetNode<GpuParticles2D>("SpineSword/SwordBone/ScaleContainer/ChargeParticles");
		_spikeParticles = GetNode<GpuParticles2D>("SpineSword/SwordBone/ScaleContainer/Spikes");
		_spikeParticles2 = GetNode<GpuParticles2D>("SpineSword/SwordBone/ScaleContainer/Spikes2");
		_spikeCircle = GetNode<GpuParticles2D>("SpineSword/SwordBone/ScaleContainer/SpikeCircle");
		_spikeCircle2 = GetNode<GpuParticles2D>("SpineSword/SwordBone/ScaleContainer/SpikeCircle2");
		_hilt = GetNode<TextureRect>("SpineSword/SwordBone/ScaleContainer/Hilt");
		_hilt2 = GetNode<TextureRect>("SpineSword/SwordBone/ScaleContainer/Hilt2");
		_detail = GetNode<TextureRect>("SpineSword/SwordBone/ScaleContainer/Detail");
		_trail = GetNode<Line2D>("Trail");
		_orbitPath = GetNode<Path2D>("%Path");
		_hitbox = GetNode<Control>("%Hitbox");
		_selectionReticle = GetNode<NSelectionReticle>("%SelectionReticle");
		_hitbox.Connect(Control.SignalName.MouseEntered, Callable.From(OnFocused));
		_hitbox.Connect(Control.SignalName.MouseExited, Callable.From(OnUnfocused));
		_hitbox.Connect(Control.SignalName.FocusEntered, Callable.From(OnFocused));
		_hitbox.Connect(Control.SignalName.FocusExited, Callable.From(OnUnfocused));
		_forgeSparks.Emitting = false;
		_forgeSparks.OneShot = true;
		_spawnFlames.Emitting = false;
		_spawnFlames.OneShot = true;
		_spawnFlamesBack.Emitting = false;
		_spawnFlamesBack.OneShot = true;
		_slashParticles.Emitting = false;
		_slashParticles.OneShot = true;
		_chargeParticles.Emitting = false;
		_spikeParticles2.Emitting = false;
		_spikeCircle2.Emitting = false;
		_bladeGlow.Modulate = Colors.Transparent;
		_bladeGlow.Visible = false;
		_trail.GlobalPosition = Vector2.Zero;
		_trail.ClearPoints();
		_animController.GetAnimationState().SetAnimation("idle_loop");
		_spineNode.Scale = Vector2.Zero;
		_spineNode.Visible = true;
		NTargetManager.Instance.Connect(NTargetManager.SignalName.TargetingBegan, Callable.From(OnTargetingBegan));
		NTargetManager.Instance.Connect(NTargetManager.SignalName.TargetingEnded, Callable.From(OnTargetingEnded));
		_owner = Card.Owner;
		_owner.Creature.Died += OnOwnerDied;
	}

	public override void _ExitTree()
	{
		_attackTween?.Kill();
		_scaleTween?.Kill();
		_sparkDelay?.Kill();
		_glowTween?.Kill();
		_owner.Creature.Died -= OnOwnerDied;
	}

	public static NSovereignBladeVfx? Create(CardModel card)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NSovereignBladeVfx nSovereignBladeVfx = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NSovereignBladeVfx>(PackedScene.GenEditState.Disabled);
		nSovereignBladeVfx.Card = card;
		return nSovereignBladeVfx;
	}

	public override void _Process(double delta)
	{
		float bakedLength = _orbitPath.Curve.GetBakedLength();
		if (_hoverTip == null)
		{
			OrbitProgress += 60.0 * delta / (double)bakedLength;
		}
		double num = OrbitProgress % 1.0;
		bool flag = num > 0.25 && num < 0.7799999713897705;
		if (flag != _isBehindCharacter && _bladeSize < 0.6f)
		{
			_isBehindCharacter = !_isBehindCharacter;
			GetParent().MoveChild(this, (!flag) ? (GetParent().GetChildCount() - 1) : 0);
		}
		Transform2D transform2D = _orbitPath.Curve.SampleBakedWithRotation((float)(OrbitProgress % 1.0) * bakedLength);
		Vector2 vector = _orbitPath.GlobalTransform * transform2D.Origin;
		vector.X = Mathf.Lerp(vector.X, base.GlobalPosition.X + 200f, Mathf.Clamp(_bladeSize / 1.25f, 0f, 1f));
		_targetOrbitPosition = vector + Vector2.Up * (_spineNode.Scale.Y - 1f) * 100f;
		if (!_isAttacking)
		{
			_spineNode.GlobalPosition = _spineNode.GlobalPosition.Lerp(_targetOrbitPosition, (float)delta * 7f);
		}
	}

	public void Forge(float bladeDamage = 0f, bool showFlames = false)
	{
		if (_isForging)
		{
			CleanupForge();
		}
		_bladeSize = Mathf.Clamp(Mathf.Lerp(0f, 1f, bladeDamage / 200f), 0f, 1f);
		_isForging = true;
		int num = (int)(_bladeSize * 30f);
		if (num > 0)
		{
			_chargeParticles.Amount = num;
			_chargeParticles.Emitting = true;
		}
		else
		{
			_chargeParticles.Emitting = false;
		}
		_hilt.Visible = _bladeSize < 0.3f;
		_hilt2.Visible = !_hilt.Visible;
		GpuParticles2D spikeParticles = _spikeParticles;
		bool emitting = (_spikeParticles.Visible = _hilt.Visible);
		spikeParticles.Emitting = emitting;
		GpuParticles2D spikeParticles2 = _spikeParticles2;
		emitting = (_spikeParticles2.Visible = !_hilt.Visible);
		spikeParticles2.Emitting = emitting;
		GpuParticles2D spikeCircle = _spikeCircle;
		emitting = (_spikeCircle.Visible = _hilt.Visible);
		spikeCircle.Emitting = emitting;
		GpuParticles2D spikeCircle2 = _spikeCircle2;
		emitting = (_spikeCircle2.Visible = !_hilt.Visible);
		spikeCircle2.Emitting = emitting;
		_detail.Visible = bladeDamage >= 0.66f;
		_bladeGlow.Visible = true;
		Color color = Color.FromHtml("#ff7300");
		Color color2 = color;
		color2.A = 0f;
		_glowTween = CreateTween();
		if (showFlames)
		{
			FireFlames();
		}
		_glowTween.TweenProperty(_bladeGlow, "modulate", color, 0.05).SetEase(Tween.EaseType.Out);
		_glowTween.Chain().TweenProperty(_bladeGlow, "modulate", color2, 0.5).SetEase(Tween.EaseType.In)
			.SetTrans(Tween.TransitionType.Cubic);
		_glowTween.Chain().TweenCallback(Callable.From(CleanupForge));
		Vector2 vector = Vector2.One * Mathf.Lerp(0.9f, 2f, _bladeSize);
		_scaleTween = CreateTween();
		_scaleTween.TweenProperty(_spineNode, "scale", vector * 1.2f, 0.05000000074505806).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		_scaleTween.Chain().TweenCallback(Callable.From(FireSparks));
		_scaleTween.Chain().TweenProperty(_spineNode, "scale", vector, 0.30000001192092896).SetEase(Tween.EaseType.InOut)
			.SetTrans(Tween.TransitionType.Cubic);
	}

	public void Attack(Vector2 targetPos)
	{
		if (_isAttacking)
		{
			CleanupAttack();
		}
		_isAttacking = true;
		_animController.GetAnimationState().SetAnimation("attack", loop: false);
		_attackTween = CreateTween();
		Vector2 vector = new Vector2(_spineNode.GlobalPosition.X - 50f, _spineNode.GlobalPosition.Y);
		_trail.Visible = true;
		_trailStart = vector;
		_attackTween.TweenProperty(_spineNode, "rotation", _spineNode.GetAngleTo(targetPos), 0.05000000074505806);
		_attackTween.Parallel().TweenProperty(_spineNode, "global_position", vector, 0.07999999821186066).SetEase(Tween.EaseType.Out)
			.SetTrans(Tween.TransitionType.Expo);
		_attackTween.Chain().TweenProperty(_spineNode, "rotation", _spineNode.GetAngleTo(targetPos), 0.0);
		_attackTween.Parallel().TweenProperty(_spineNode, "global_position", targetPos, 0.05000000074505806).SetEase(Tween.EaseType.In)
			.SetTrans(Tween.TransitionType.Expo);
		_attackTween.Chain().TweenCallback(Callable.From(EndSlash));
		_attackTween.TweenInterval(0.25);
		_attackTween.Chain().TweenCallback(Callable.From(FireSparks)).SetDelay(0.30000001192092896);
		_attackTween.Chain().TweenCallback(Callable.From(CleanupAttack));
		UpdateHoverTip();
	}

	private void OnTargetingBegan()
	{
		_hitbox.MouseFilter = Control.MouseFilterEnum.Ignore;
		UpdateHoverTip();
	}

	private void OnTargetingEnded()
	{
		_hitbox.MouseFilter = Control.MouseFilterEnum.Stop;
		UpdateHoverTip();
	}

	private void OnFocused()
	{
		_isFocused = true;
		if (!NCombatRoom.Instance.Ui.Hand.InCardPlay)
		{
			UpdateHoverTip();
		}
	}

	private void OnUnfocused()
	{
		_isFocused = false;
		UpdateHoverTip();
	}

	private void UpdateHoverTip()
	{
		bool flag = _isFocused && !_isAttacking && !NTargetManager.Instance.IsInSelection && _hitbox.MouseFilter != Control.MouseFilterEnum.Ignore;
		if (flag && _hoverTip == null)
		{
			_hoverTip = NHoverTipSet.CreateAndShow(_hitbox, HoverTipFactory.FromCard(Card));
			_hoverTip.GlobalPosition = _hitbox.GlobalPosition + Vector2.Right * _hitbox.Size.X;
			_selectionReticle.OnSelect();
		}
		else if (!flag && _hoverTip != null)
		{
			NHoverTipSet.Remove(_hitbox);
			_selectionReticle.OnDeselect();
			_hoverTip = null;
		}
	}

	private void FireSparks()
	{
		_forgeSparks.Restart();
	}

	private void FireFlames()
	{
		_spawnFlames.Restart();
		_spawnFlamesBack.Restart();
	}

	private void EndSlash()
	{
		_chargeParticles.Emitting = false;
		_chargeParticles.Restart();
		_slashParticles.Rotation = _spineNode.GetAngleTo(_trailStart) - 1.5708f;
		_slashParticles.Restart();
		_trail.AddPoint(_trailStart);
		_trail.AddPoint(GetNode<Node2D>("SpineSword/SwordBone/ScaleContainer/SpikeCircle").GlobalPosition);
		_trail.Modulate = Colors.White;
		CreateTween().TweenProperty(_trail, "modulate:a", 0f, 0.20000000298023224);
	}

	private void CleanupForge()
	{
		_isForging = false;
		_scaleTween?.Kill();
		_glowTween?.Kill();
	}

	private void CleanupAttack()
	{
		_isAttacking = false;
		_attackTween?.Kill();
		_animController.GetAnimationState().SetAnimation("idle_loop");
		_spineNode.Rotation = 0f;
		_trail.ClearPoints();
	}

	public void RemoveSovereignBlade()
	{
		_scaleTween?.Kill();
		_scaleTween = CreateTween();
		_scaleTween.TweenProperty(_spineNode, "scale", Vector2.Zero, 0.20000000298023224).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		_scaleTween.Chain().TweenCallback(Callable.From(this.QueueFreeSafely));
	}

	private void OnOwnerDied(Creature creature)
	{
		_hitbox.MouseFilter = Control.MouseFilterEnum.Ignore;
		UpdateHoverTip();
		RemoveSovereignBlade();
	}
}
