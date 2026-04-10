using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereItems;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;

public partial class NCrystalSphereItem : Control
{
	public const string scenePath = "res://scenes/events/custom/crystal_sphere/crystal_sphere_item.tscn";

	private CrystalSphereItem _item;

	private TextureRect _icon;

	private Material _material;

	private Control _card;

	private TextureRect _cardFrame;

	private TextureRect _cardBanner;

	private Tween? _tween;

	public static NCrystalSphereItem? Create(CrystalSphereItem item)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCrystalSphereItem nCrystalSphereItem = PreloadManager.Cache.GetScene("res://scenes/events/custom/crystal_sphere/crystal_sphere_item.tscn").Instantiate<NCrystalSphereItem>(PackedScene.GenEditState.Disabled);
		nCrystalSphereItem._item = item;
		return nCrystalSphereItem;
	}

	public override void _Ready()
	{
		_icon = GetNode<TextureRect>("Icon");
		_card = GetNode<Control>("%Card");
		_cardFrame = GetNode<TextureRect>("%CardFrame");
		_cardBanner = GetNode<TextureRect>("%CardBanner");
		base.PivotOffset = base.Size / 2f;
		_icon.PivotOffset = base.Size / 2f;
		_card.PivotOffset = base.Size / 2f;
		_material = _icon.Material;
		if (_item is CrystalSphereCardReward crystalSphereCardReward)
		{
			_card.Visible = true;
			_icon.Visible = false;
			_cardBanner.Material = crystalSphereCardReward.BannerMaterial;
			_cardFrame.Material = crystalSphereCardReward.FrameMaterial;
		}
		else
		{
			_card.Visible = false;
			_icon.Visible = true;
			_icon.Texture = _item.Texture;
		}
	}

	public override void _EnterTree()
	{
		_item.Revealed += OnRevealed;
	}

	public override void _ExitTree()
	{
		_item.Revealed -= OnRevealed;
		if (_tween != null && _tween.IsRunning())
		{
			_tween.Kill();
		}
	}

	private void OnRevealed(CrystalSphereItem item)
	{
		_tween?.Kill();
		_tween = CreateTween();
		_tween.TweenInterval(0.25);
		_tween.Chain().TweenProperty(_material, "shader_parameter/val", 1f, 0.5).From(0);
		_tween.Parallel().TweenProperty(_icon, "scale", Vector2.One * 1.2f, 0.15000000596046448);
		_tween.Parallel().TweenProperty(_icon, "scale", Vector2.One, 0.5).SetDelay(0.15000000596046448);
	}
}
