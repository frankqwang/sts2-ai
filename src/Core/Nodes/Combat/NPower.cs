using System;
using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Powers;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Rooms;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NPower : Control
{
	private static readonly StringName _pulse = new StringName("pulse");

	private PowerModel? _model;

	private TextureRect _icon;

	private MegaLabel _amountLabel;

	private CpuParticles2D _powerFlash;

	private Tween? _animInTween;

	public NPowerContainer Container { get; set; }

	private static string ScenePath => SceneHelper.GetScenePath("combat/power");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	public PowerModel Model
	{
		get
		{
			return _model ?? throw new InvalidOperationException("Model was accessed before it was set.");
		}
		set
		{
			if (_model != null)
			{
				UnsubscribeFromModelEvents();
			}
			value.AssertMutable();
			_model = value;
			if (_model != null && IsInsideTree())
			{
				SubscribeToModelEvents();
			}
			Reload();
		}
	}

	public static NPower Create(PowerModel power)
	{
		NPower nPower = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NPower>(PackedScene.GenEditState.Disabled);
		nPower.Model = power;
		return nPower;
	}

	public override void _Ready()
	{
		_icon = GetNode<TextureRect>("%Icon");
		_amountLabel = GetNode<MegaLabel>("%AmountLabel");
		_powerFlash = GetNode<CpuParticles2D>("%PowerFlash");
		Connect(Control.SignalName.MouseEntered, Callable.From(OnHovered));
		Connect(Control.SignalName.MouseExited, Callable.From(OnUnhovered));
		Reload();
		_animInTween?.Kill();
		_animInTween = CreateTween().SetParallel();
		_animInTween.TweenProperty(_icon, "position:y", 0f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back)
			.From(-24f);
		_animInTween.TweenProperty(this, "modulate:a", 1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	public override void _EnterTree()
	{
		SubscribeToModelEvents();
	}

	public override void _ExitTree()
	{
		UnsubscribeFromModelEvents();
	}

	private void Reload()
	{
		if (IsNodeReady())
		{
			_icon.Texture = _model?.Icon;
			_powerFlash.Texture = _model?.BigIcon;
			RefreshAmount();
		}
	}

	private void OnPulsingStarted()
	{
		ShaderMaterial shaderMaterial = (ShaderMaterial)_icon.Material;
		shaderMaterial.SetShaderParameter(_pulse, 1);
	}

	private void OnPulsingStopped()
	{
		ShaderMaterial shaderMaterial = (ShaderMaterial)_icon.Material;
		shaderMaterial.SetShaderParameter(_pulse, 0);
	}

	private void RefreshAmount()
	{
		if (_model != null)
		{
			_amountLabel.AddThemeColorOverride(ThemeConstants.Label.fontColor, Model.AmountLabelColor);
			_amountLabel.SetTextAutoSize((Model.StackType == PowerStackType.Counter) ? Model.DisplayAmount.ToString() : string.Empty);
		}
		else
		{
			_amountLabel.SetTextAutoSize(string.Empty);
		}
	}

	private void OnDisplayAmountChanged()
	{
		FlashPower();
		RefreshAmount();
	}

	private void OnPowerFlashed(PowerModel _)
	{
		FlashPower();
	}

	private void FlashPower()
	{
		_powerFlash.Emitting = true;
	}

	private void OnHovered()
	{
		NCombatRoom.Instance?.GetCreatureNode(Model.Owner)?.ShowHoverTips(Model.HoverTips);
		_icon.Scale = Vector2.One * 1.1f;
		CombatManager.Instance.StateTracker.CombatStateChanged += ShowPowerHoverTips;
	}

	private void OnUnhovered()
	{
		NCombatRoom.Instance?.GetCreatureNode(Model.Owner)?.HideHoverTips();
		_icon.Scale = Vector2.One * 1f;
		CombatManager.Instance.StateTracker.CombatStateChanged -= ShowPowerHoverTips;
	}

	private void ShowPowerHoverTips(CombatState _)
	{
		NCombatRoom.Instance?.GetCreatureNode(Model.Owner)?.ShowHoverTips(Model.HoverTips);
	}

	private void SubscribeToModelEvents()
	{
		if (_model != null)
		{
			Model.DisplayAmountChanged += OnDisplayAmountChanged;
			Model.Flashed += OnPowerFlashed;
			Model.Removed += OnPowerRemoved;
			Model.Owner.Died += OnOwnerDied;
			Model.Owner.Revived += OnOwnerRevived;
			Model.PulsingStarted += OnPulsingStarted;
			Model.PulsingStopped += OnPulsingStopped;
		}
	}

	private void UnsubscribeFromModelEvents()
	{
		if (_model != null)
		{
			Model.DisplayAmountChanged -= OnDisplayAmountChanged;
			Model.Flashed -= OnPowerFlashed;
			Model.Removed -= OnPowerRemoved;
			Model.Owner.Died -= OnOwnerDied;
			Model.Owner.Revived -= OnOwnerRevived;
			Model.PulsingStarted -= OnPulsingStarted;
			Model.PulsingStopped -= OnPulsingStopped;
		}
	}

	private void OnPowerRemoved()
	{
		UnsubscribeFromModelEvents();
	}

	private void OnOwnerDied(Creature _)
	{
		if (GodotObject.IsInstanceValid(this) && Model.ShouldPowerBeRemovedAfterOwnerDeath())
		{
			base.MouseFilter = MouseFilterEnum.Ignore;
		}
	}

	private void OnOwnerRevived(Creature _)
	{
		base.MouseFilter = MouseFilterEnum.Stop;
	}
}
