using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;
using MegaCrit.Sts2.Core.Random;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NDecimillipedeSegmentVfx : Node
{
	private static readonly StringName _opacity = new StringName("opacity");

	private static readonly StringName _direction = new StringName("direction");

	[Export(PropertyHint.None, "")]
	private CpuParticles2D[] _damageParticleNodes;

	private readonly Vector2 _particleGravity = new Vector2(0f, 300f);

	private float _particleSpeedScale = 2f;

	private Vector2 _particleVelocityMinMax = new Vector2(400f, 600f);

	[Export(PropertyHint.None, "")]
	private Node2D[] _sprayNodes;

	private Node2D _parent;

	private MegaSprite _animController;

	private readonly List<Vector2> _sprayNodeScales = new List<Vector2>();

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		Node2D[] sprayNodes = _sprayNodes;
		foreach (Node2D node2D in sprayNodes)
		{
			node2D.Visible = false;
			_sprayNodeScales.Add(node2D.Scale);
		}
	}

	public void Regenerate()
	{
		for (int i = 0; i < _sprayNodes.Length; i++)
		{
			Node2D node2D = _sprayNodes[i];
			node2D.Visible = true;
			ShaderMaterial shaderMaterial = (ShaderMaterial)node2D.Material;
			shaderMaterial.SetShaderParameter(_direction, -1);
			Tween tween = CreateTween().SetParallel();
			float num = Rng.Chaotic.NextFloat(0.5f);
			shaderMaterial.SetShaderParameter(_opacity, 0.5f);
			tween.TweenProperty(node2D, "scale", _sprayNodeScales[i], num + 0.5f).SetTrans(Tween.TransitionType.Expo).SetEase(Tween.EaseType.Out);
			tween.TweenProperty(node2D.Material, "shader_parameter/opacity", 0.9f, num).SetTrans(Tween.TransitionType.Expo).SetEase(Tween.EaseType.Out);
		}
	}

	private void EndRegenerate()
	{
		Node2D[] sprayNodes = _sprayNodes;
		foreach (Node2D node2D in sprayNodes)
		{
			Vector2 scale = node2D.Scale;
			Tween tween = CreateTween();
			float num = Rng.Chaotic.NextFloat(0.9f, 1.25f);
			tween.TweenProperty(node2D, "scale", Vector2.Zero, num).SetTrans(Tween.TransitionType.Quad).SetEase(Tween.EaseType.In);
			tween.TweenProperty(node2D, "visible", false, 0.0);
			tween.TweenProperty(node2D, "scale", scale, 0.0);
		}
	}

	private void Wither()
	{
		CpuParticles2D[] damageParticleNodes = _damageParticleNodes;
		foreach (CpuParticles2D cpuParticles2D in damageParticleNodes)
		{
			cpuParticles2D.Gravity = _particleGravity;
			cpuParticles2D.SpeedScale = _particleSpeedScale;
			cpuParticles2D.InitialVelocityMin = _particleVelocityMinMax.X;
			cpuParticles2D.InitialVelocityMax = _particleVelocityMinMax.Y;
			cpuParticles2D.Restart();
		}
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject animEvent)
	{
		string eventName = new MegaEvent(animEvent).GetData().GetEventName();
		if (!(eventName == "suck_complete"))
		{
			if (eventName == "explode")
			{
				Wither();
			}
		}
		else
		{
			EndRegenerate();
		}
	}
}
