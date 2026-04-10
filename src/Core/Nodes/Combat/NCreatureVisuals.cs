using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NCreatureVisuals : Node2D
{
	private static readonly StringName _overlayInfluence = new StringName("overlay_influence");

	private static readonly StringName _h = new StringName("h");

	private static readonly StringName _tint = new StringName("tint");

	private const double _baseLiquidOverlayDuration = 1.0;

	private float _hue = 1f;

	private double _liquidOverlayTimer;

	private Material? _savedNormalMaterial;

	private ShaderMaterial? _currentLiquidOverlayMaterial;

	public Node2D Body { get; private set; }

	public Control Bounds { get; private set; }

	public Marker2D IntentPosition { get; private set; }

	public Marker2D OrbPosition { get; private set; }

	public Marker2D? TalkPosition { get; private set; }

	public bool HasSpineAnimation
	{
		get
		{
			if (GodotObject.IsInstanceValid(Body))
			{
				return Body.GetClass() == "SpineSprite";
			}
			return false;
		}
	}

	public MegaSprite? SpineBody { get; private set; }

	public Marker2D VfxSpawnPosition { get; private set; }

	public float DefaultScale { get; set; } = 1f;

	public override void _Ready()
	{
		Body = GetNode<Node2D>("%Visuals");
		Bounds = GetNode<Control>("%Bounds");
		IntentPosition = GetNode<Marker2D>("%IntentPos");
		VfxSpawnPosition = GetNode<Marker2D>("%CenterPos");
		OrbPosition = (HasNode("%OrbPos") ? GetNode<Marker2D>("%OrbPos") : IntentPosition);
		TalkPosition = (HasNode("%TalkPos") ? GetNode<Marker2D>("%TalkPos") : null);
		if (HasSpineAnimation)
		{
			SpineBody = new MegaSprite(Body);
		}
		_savedNormalMaterial = null;
		_currentLiquidOverlayMaterial = null;
	}

	public void SetUpSkin(MonsterModel model)
	{
		if (SpineBody?.GetSkeleton() != null)
		{
			model.SetupSkins(this);
		}
	}

	public void SetScaleAndHue(float scale, float hue)
	{
		DefaultScale = scale;
		base.Scale = Vector2.One * scale;
		_hue = hue;
		if (!Mathf.IsEqualApprox(hue, 0f) && SpineBody != null)
		{
			Material normalMaterial = SpineBody.GetNormalMaterial();
			ShaderMaterial shaderMaterial;
			if (normalMaterial == null)
			{
				Material material = (ShaderMaterial)PreloadManager.Cache.GetMaterial("res://materials/vfx/hsv.tres");
				shaderMaterial = (ShaderMaterial)material.Duplicate();
				SpineBody.SetNormalMaterial(shaderMaterial);
			}
			else
			{
				shaderMaterial = (ShaderMaterial)normalMaterial;
			}
			shaderMaterial.SetShaderParameter(_h, hue);
		}
	}

	public bool IsPlayingHurtAnimation()
	{
		if (SpineBody?.GetSkeleton() != null)
		{
			return SpineBody.GetAnimationState().GetCurrent(0).GetAnimation()
				.GetName()
				.Equals("hurt");
		}
		return false;
	}

	public void TryApplyLiquidOverlay(Color tint)
	{
		if (_currentLiquidOverlayMaterial != null)
		{
			_currentLiquidOverlayMaterial.SetShaderParameter(_tint, tint);
			_liquidOverlayTimer = 1.0;
		}
		else
		{
			TaskHelper.RunSafely(ApplyLiquidOverlayInternal(tint));
		}
	}

	private async Task ApplyLiquidOverlayInternal(Color tint)
	{
		if (SpineBody != null)
		{
			_savedNormalMaterial = SpineBody.GetNormalMaterial();
			Material material = (ShaderMaterial)PreloadManager.Cache.GetMaterial("res://materials/vfx/potion/potion_liquid_overlay.tres");
			_currentLiquidOverlayMaterial = (ShaderMaterial)material.Duplicate();
			_currentLiquidOverlayMaterial.SetShaderParameter(_tint, tint);
			_currentLiquidOverlayMaterial.SetShaderParameter(_h, _hue);
			_currentLiquidOverlayMaterial.SetShaderParameter(_overlayInfluence, 1f);
			SpineBody.SetNormalMaterial(_currentLiquidOverlayMaterial);
			_liquidOverlayTimer = 1.0;
			while (_liquidOverlayTimer > 0.0)
			{
				double num = (1.0 - _liquidOverlayTimer) / 1.0;
				_currentLiquidOverlayMaterial.SetShaderParameter(_overlayInfluence, 1.0 - num);
				_liquidOverlayTimer -= GetProcessDeltaTime();
				await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
			}
			SpineBody.SetNormalMaterial(_savedNormalMaterial);
			_currentLiquidOverlayMaterial = null;
		}
	}
}
