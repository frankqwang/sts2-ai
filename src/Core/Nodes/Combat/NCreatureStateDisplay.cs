using System;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.UI;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NCreatureStateDisplay : Control
{
	private NPowerContainer _powerContainer;

	private Control _nameplateContainer;

	private MegaLabel _nameplateLabel;

	private NHealthBar _healthBar;

	private Control _hpBarHitbox;

	private Creature? _creature;

	private Vector2 _creatureSize;

	private Creature? _blockTrackingCreature;

	private Tween? _showHideTween;

	private Tween? _hoverTween;

	private static readonly Vector2 _healthBarAnimOffset = new Vector2(0f, 20f);

	private Vector2 _originalPosition;

	public override void _Ready()
	{
		_powerContainer = GetNode<NPowerContainer>("%PowerContainer");
		_nameplateContainer = GetNode<Control>("%NameplateContainer");
		_nameplateLabel = GetNode<MegaLabel>("%NameplateLabel");
		_healthBar = GetNode<NHealthBar>("%HealthBar");
		_hpBarHitbox = GetNode<Control>("%HpBarHitbox");
		_nameplateContainer.Modulate = StsColors.transparentWhite;
		_originalPosition = base.Position;
		_hpBarHitbox.Connect(Control.SignalName.MouseEntered, Callable.From(OnHovered));
		_hpBarHitbox.Connect(Control.SignalName.MouseExited, Callable.From(OnUnhovered));
	}

	public override void _EnterTree()
	{
		base._EnterTree();
		CombatManager.Instance.StateTracker.CombatStateChanged += OnCombatStateChanged;
		SubscribeToCreatureEvents();
		if (NCombatRoom.Instance != null)
		{
			NCombatRoom.Instance.Ui.DebugToggleHpBar += DebugToggleVisibility;
		}
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		CombatManager.Instance.StateTracker.CombatStateChanged -= OnCombatStateChanged;
		if (_creature != null)
		{
			_creature.BlockChanged -= AnimateInBlock;
			_creature.Died -= OnCreatureDied;
			_creature.Revived -= OnCreatureRevived;
		}
		if (_blockTrackingCreature != null)
		{
			_blockTrackingCreature.BlockChanged -= OnBlockTrackingCreatureBlockChanged;
		}
		if (NCombatRoom.Instance != null)
		{
			NCombatRoom.Instance.Ui.DebugToggleHpBar -= DebugToggleVisibility;
		}
	}

	public void SetCreature(Creature creature)
	{
		if (_creature != null)
		{
			throw new InvalidOperationException("Creature was already set.");
		}
		_creature = creature;
		SubscribeToCreatureEvents();
		_nameplateLabel.SetTextAutoSize(creature.Name);
		_powerContainer.SetCreature(_creature);
		_healthBar.SetCreature(_creature);
		RefreshValues();
	}

	private void SubscribeToCreatureEvents()
	{
		if (_creature != null)
		{
			_creature.BlockChanged += AnimateInBlock;
			_creature.Died += OnCreatureDied;
			_creature.Revived += OnCreatureRevived;
		}
	}

	private void DebugToggleVisibility()
	{
		base.Visible = !NCombatUi.IsDebugHidingHpBar;
	}

	public void SetCreatureBounds(Control bounds)
	{
		_healthBar.UpdateLayoutForCreatureBounds(bounds);
		_nameplateContainer.GlobalPosition = new Vector2(bounds.GlobalPosition.X, _nameplateContainer.GlobalPosition.Y);
		_nameplateContainer.Size = new Vector2(bounds.Size.X * bounds.Scale.X, _nameplateContainer.Size.Y);
		_powerContainer.SetCreatureBounds(bounds);
		RefreshValues();
	}

	private void RefreshValues()
	{
		if (_creature != null)
		{
			_nameplateLabel.SetTextAutoSize(_creature.Name);
			_healthBar.RefreshValues();
		}
	}

	private void OnCombatStateChanged(CombatState _)
	{
		RefreshValues();
	}

	private void OnHovered()
	{
		_healthBar.FadeOutHpLabel(0.5f, 0.1f);
		ShowNameplate();
		if (!NTargetManager.Instance.IsInSelection)
		{
			NCombatRoom.Instance?.GetCreatureNode(_creature)?.ShowHoverTips(_creature.HoverTips);
		}
	}

	private void OnUnhovered()
	{
		_healthBar.FadeInHpLabel(0.5f);
		HideNameplate();
		NCombatRoom.Instance.GetCreatureNode(_creature)?.HideHoverTips();
	}

	public void ShowNameplate()
	{
		_hoverTween?.Kill();
		_hoverTween = CreateTween().SetParallel();
		_hoverTween.TweenProperty(_powerContainer, "modulate:a", 0.5f, 0.15000000596046448);
		_hoverTween.TweenProperty(_nameplateContainer, "modulate:a", 1, 0.15000000596046448);
	}

	public void HideNameplate()
	{
		_hoverTween?.Kill();
		_hoverTween = CreateTween().SetParallel();
		_hoverTween.TweenProperty(_powerContainer, "modulate:a", 1f, 0.20000000298023224);
		_hoverTween.TweenProperty(_nameplateContainer, "modulate:a", 0, 0.20000000298023224);
	}

	public void HideImmediately()
	{
		Color modulate = base.Modulate;
		modulate.A = 0f;
		base.Modulate = modulate;
	}

	public void AnimateIn(HealthBarAnimMode mode)
	{
		if (SaveManager.Instance.PrefsSave.FastMode == FastModeType.Instant)
		{
			Color modulate = base.Modulate;
			modulate.A = 1f;
			base.Modulate = modulate;
			base.Visible = true;
			return;
		}
		float num = 0f;
		base.Visible = true;
		base.Modulate = StsColors.transparentWhite;
		base.Position -= _healthBarAnimOffset;
		if (mode == HealthBarAnimMode.SpawnedAtCombatStart)
		{
			num = Rng.Chaotic.NextFloat(1.3f, 1.7f);
		}
		_showHideTween?.Kill();
		_showHideTween = CreateTween().SetParallel();
		_showHideTween.TweenProperty(this, "modulate:a", 1f, (mode == HealthBarAnimMode.FromHidden) ? 0.15f : 1f).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Sine)
			.SetDelay(num);
		_showHideTween.TweenProperty(this, "position", _originalPosition, (mode == HealthBarAnimMode.FromHidden) ? 0.15f : 0.5f).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Quad)
			.SetDelay(num);
	}

	private void AnimateInBlock(int oldBlock, int blockGain)
	{
		if (oldBlock == 0 && blockGain != 0)
		{
			_healthBar.AnimateInBlock(oldBlock, blockGain);
		}
	}

	public void AnimateOut()
	{
		if (SaveManager.Instance.PrefsSave.FastMode == FastModeType.Instant)
		{
			Color modulate = base.Modulate;
			modulate.A = 0f;
			base.Modulate = modulate;
			base.Visible = false;
			return;
		}
		_showHideTween?.Kill();
		_showHideTween = CreateTween().SetParallel();
		_showHideTween.TweenProperty(this, "modulate:a", 0f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Sine);
		_showHideTween.TweenProperty(this, "position", _healthBarAnimOffset, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Quad);
		_showHideTween.Chain().TweenCallback(Callable.From(() => base.Visible = false));
	}

	private void OnCreatureDied(Creature _)
	{
		_hpBarHitbox.MouseFilter = MouseFilterEnum.Ignore;
	}

	private void OnCreatureRevived(Creature _)
	{
		_hpBarHitbox.MouseFilter = MouseFilterEnum.Stop;
	}

	public void TrackBlockStatus(Creature creature)
	{
		_blockTrackingCreature = creature;
		_blockTrackingCreature.BlockChanged += OnBlockTrackingCreatureBlockChanged;
		_healthBar.TrackBlockStatus(creature);
	}

	private void OnBlockTrackingCreatureBlockChanged(int oldBlock, int blockGain)
	{
		_healthBar.RefreshValues();
	}
}
