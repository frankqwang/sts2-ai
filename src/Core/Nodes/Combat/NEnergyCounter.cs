using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NEnergyCounter : Control
{
	private const string _darkenedMatPath = "res://materials/ui/energy_orb_dark.tres";

	private Player _player;

	private MegaLabel _label;

	private Control _layers;

	private Control _rotationLayers;

	private CpuParticles2D _backParticles;

	private CpuParticles2D _frontParticles;

	private HoverTip _hoverTip;

	private Tween? _animInTween;

	private Tween? _animOutTween;

	private const float _animDuration = 0.6f;

	private static readonly Vector2 _showPosition = Vector2.Zero;

	private static readonly Vector2 _hidePosition = new Vector2(-480f, 128f);

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>("res://materials/ui/energy_orb_dark.tres");

	private Color OutlineColor => _player.Character.EnergyLabelOutlineColor;

	public static NEnergyCounter? Create(Player player)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NEnergyCounter nEnergyCounter = PreloadManager.Cache.GetScene(player.Character.EnergyCounterPath).Instantiate<NEnergyCounter>(PackedScene.GenEditState.Disabled);
		nEnergyCounter._player = player;
		return nEnergyCounter;
	}

	public override void _Ready()
	{
		_label = GetNode<MegaLabel>("Label");
		_layers = GetNode<Control>("%Layers");
		_rotationLayers = GetNode<Control>("%RotationLayers");
		_backParticles = GetNode<CpuParticles2D>("%BurstBack");
		_frontParticles = GetNode<CpuParticles2D>("%BurstFront");
		LocString locString = new LocString("static_hover_tips", "ENERGY_COUNT.description");
		locString.Add("energyPrefix", EnergyIconHelper.GetPrefix(_player.Character.CardPool));
		_hoverTip = new HoverTip(new LocString("static_hover_tips", "ENERGY_COUNT.title"), locString);
		Connect(Control.SignalName.MouseEntered, Callable.From(OnHovered));
		Connect(Control.SignalName.MouseExited, Callable.From(OnUnhovered));
		RefreshLabel();
	}

	public override void _EnterTree()
	{
		base._EnterTree();
		CombatManager.Instance.StateTracker.CombatStateChanged += OnCombatStateChanged;
		_player.PlayerCombatState.EnergyChanged += OnEnergyChanged;
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		CombatManager.Instance.StateTracker.CombatStateChanged -= OnCombatStateChanged;
		_player.PlayerCombatState.EnergyChanged -= OnEnergyChanged;
	}

	private void OnHovered()
	{
		NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, _hoverTip);
		nHoverTipSet.GlobalPosition = base.GlobalPosition + new Vector2(-70f, -200f);
	}

	private void OnUnhovered()
	{
		NHoverTipSet.Remove(this);
	}

	private void OnCombatStateChanged(CombatState combatState)
	{
		RefreshLabel();
	}

	private void RefreshLabel()
	{
		PlayerCombatState playerCombatState = _player.PlayerCombatState;
		_label.SetTextAutoSize($"{playerCombatState.Energy}/{playerCombatState.MaxEnergy}");
		_label.AddThemeColorOverride(ThemeConstants.Label.fontColor, (playerCombatState.Energy == 0) ? StsColors.red : StsColors.cream);
		_label.AddThemeColorOverride(ThemeConstants.Label.fontOutlineColor, (playerCombatState.Energy == 0) ? StsColors.unplayableEnergyCostOutline : OutlineColor);
		Material material = ((playerCombatState.Energy == 0) ? PreloadManager.Cache.GetMaterial("res://materials/ui/energy_orb_dark.tres") : null);
		foreach (Control item in _layers.GetChildren().OfType<Control>())
		{
			item.Material = material;
		}
		foreach (Control item2 in _rotationLayers.GetChildren().OfType<Control>())
		{
			item2.Material = material;
		}
		_layers.Modulate = ((playerCombatState.Energy == 0) ? Colors.DarkGray : Colors.White);
	}

	private void OnEnergyChanged(int oldEnergy, int newEnergy)
	{
		if (oldEnergy < newEnergy)
		{
			_frontParticles.Emitting = true;
			_backParticles.Emitting = true;
		}
	}

	public override void _Process(double delta)
	{
		float num = ((_player.PlayerCombatState.Energy == 0) ? 5f : 30f);
		for (int i = 0; i < _rotationLayers.GetChildCount(); i++)
		{
			_rotationLayers.GetChild<Control>(i).RotationDegrees += (float)delta * num * (float)(i + 1);
		}
	}

	public void AnimIn()
	{
		_animOutTween?.Kill();
		_animInTween = CreateTween();
		base.Position = _hidePosition;
		_animInTween.TweenProperty(this, "position", _showPosition, 0.6000000238418579).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	public void AnimOut()
	{
		_animInTween?.Kill();
		_animOutTween = CreateTween();
		base.Position = _showPosition;
		_animOutTween.TweenProperty(this, "position", _hidePosition, 0.6000000238418579).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Back);
	}
}
