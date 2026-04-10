using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;

public partial class NCrystalSphereCell : NClickableControl
{
	private const string _scenePath = "res://scenes/events/custom/crystal_sphere/crystal_sphere_cell.tscn";

	private NCrystalSphereMask _mask;

	private Control _hoveredFg;

	private Tween? _fadeTween;

	public CrystalSphereCell Entity { get; private set; }

	public static NCrystalSphereCell? Create(CrystalSphereCell cell, NCrystalSphereMask mask)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCrystalSphereCell nCrystalSphereCell = PreloadManager.Cache.GetScene("res://scenes/events/custom/crystal_sphere/crystal_sphere_cell.tscn").Instantiate<NCrystalSphereCell>(PackedScene.GenEditState.Disabled);
		nCrystalSphereCell.Entity = cell;
		nCrystalSphereCell._mask = mask;
		return nCrystalSphereCell;
	}

	public override void _Ready()
	{
		_hoveredFg = GetNode<Control>("HoveredFg");
		base.Modulate = Colors.Transparent;
		base.MouseFilter = (MouseFilterEnum)(Entity.IsHidden ? 0 : 2);
		base.FocusMode = (FocusModeEnum)(Entity.IsHidden ? 2 : 0);
		_hoveredFg.Visible = false;
	}

	public override void _EnterTree()
	{
		Entity.HighlightUpdated += OnEntityHighlightUpdated;
		Entity.FogUpdated += EntityClicked;
	}

	public override void _ExitTree()
	{
		Entity.HighlightUpdated -= OnEntityHighlightUpdated;
		Entity.FogUpdated -= EntityClicked;
	}

	private void OnEntityHighlightUpdated()
	{
		_fadeTween?.Kill();
		_fadeTween = CreateTween();
		Tween? fadeTween = _fadeTween;
		NodePath property = "modulate";
		CrystalSphereCell entity = Entity;
		fadeTween.TweenProperty(this, property, (entity != null && entity.IsHighlighted && entity.IsHidden) ? Colors.White : Colors.Transparent, 0.15000000596046448);
		_hoveredFg.Visible = Entity.IsHovered;
	}

	private void EntityClicked()
	{
		base.MouseFilter = (MouseFilterEnum)(Entity.IsHidden ? 0 : 2);
		base.FocusMode = (FocusModeEnum)(Entity.IsHidden ? 2 : 0);
		_mask.UpdateMat(Entity);
	}
}
