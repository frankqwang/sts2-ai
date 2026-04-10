using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NLivingGasVfx : Node
{
	private static readonly StringName _alphaStep = new StringName("AlphaStep");

	private GpuParticles2D _attackPuffParticles;

	private GpuParticles2D _attackSparkParticles;

	private GpuParticles2D _debuffPuffParticles;

	private GpuParticles2D _gasPuffParticles;

	private MegaSlotNode _smokeSlot1;

	private MegaSlotNode _smokeSlot2;

	private MegaSlotNode _smokeSlot3;

	private List<ShaderMaterial?> _smokeMaterials;

	private List<Vector2> _smokeSteps;

	private Node2D _parent;

	private MegaSprite _megaSprite;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_attackPuffParticles = GetNode<GpuParticles2D>("../AttackFXNode/AttackPuffParticles");
		_attackSparkParticles = GetNode<GpuParticles2D>("../AttackFXNode/AttackSparkParticles");
		_debuffPuffParticles = GetNode<GpuParticles2D>("../AttackFXNode/DebuffPuffParticles");
		_gasPuffParticles = GetNode<GpuParticles2D>("../AttackFXNode/GasPuffParticles");
		_smokeSlot1 = new MegaSlotNode(GetNode<Node2D>("../SmokeSlot1"));
		_smokeSlot2 = new MegaSlotNode(GetNode<Node2D>("../SmokeSlot2"));
		_smokeSlot3 = new MegaSlotNode(GetNode<Node2D>("../SmokeSlot3"));
		_smokeMaterials = new List<ShaderMaterial>();
		_smokeMaterials.Add(_smokeSlot1.GetNormalMaterial() as ShaderMaterial);
		_smokeMaterials.Add(_smokeSlot2.GetNormalMaterial() as ShaderMaterial);
		_smokeMaterials.Add(_smokeSlot3.GetNormalMaterial() as ShaderMaterial);
		_smokeSteps = new List<Vector2>();
		_smokeSteps.Add((Vector2)_smokeMaterials[0].GetShaderParameter(_alphaStep));
		_smokeSteps.Add((Vector2)_smokeMaterials[1].GetShaderParameter(_alphaStep));
		_smokeSteps.Add((Vector2)_smokeMaterials[2].GetShaderParameter(_alphaStep));
		_attackPuffParticles.Emitting = false;
		_attackSparkParticles.Emitting = false;
		_debuffPuffParticles.Emitting = false;
		_gasPuffParticles.Emitting = false;
		_megaSprite = new MegaSprite(GetParent<Node2D>());
		_megaSprite.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		string eventName = new MegaEvent(spineEvent).GetData().GetEventName();
		if (eventName == null)
		{
			return;
		}
		switch (eventName.Length)
		{
		case 12:
			switch (eventName[0])
			{
			case 'a':
				if (eventName == "attack_start")
				{
					OnAttackStart();
				}
				break;
			case 'd':
				if (eventName == "debuff_start")
				{
					OnDebuffStart();
				}
				break;
			case 'r':
				if (eventName == "reconstitute")
				{
					OnReconstitute();
				}
				break;
			}
			break;
		case 10:
			switch (eventName[0])
			{
			case 'a':
				if (eventName == "attack_end")
				{
					OnAttackEnd();
				}
				break;
			case 'h':
				if (eventName == "hurt_start")
				{
					OnHurtStart();
				}
				break;
			case 'd':
				if (eventName == "debuff_end")
				{
					OnDebuffEnd();
				}
				break;
			}
			break;
		case 8:
			if (eventName == "hurt_end")
			{
				OnHurtEnd();
			}
			break;
		case 9:
			if (eventName == "dissipate")
			{
				OnDissipate();
			}
			break;
		case 16:
			if (eventName == "gas_breath_start")
			{
				OnDeathBreathStart();
			}
			break;
		case 14:
			if (eventName == "gas_breath_end")
			{
				OnDeathBreathEnd();
			}
			break;
		case 11:
		case 13:
		case 15:
			break;
		}
	}

	private void OnAttackStart()
	{
		_attackPuffParticles.Emitting = true;
		_attackSparkParticles.Amount = 24;
		_attackSparkParticles.Restart();
	}

	private void OnAttackEnd()
	{
		_attackPuffParticles.Emitting = false;
		_attackSparkParticles.Emitting = false;
	}

	private void OnHurtStart()
	{
		_attackSparkParticles.Amount = 100;
		_attackSparkParticles.Emitting = true;
	}

	private void OnHurtEnd()
	{
		_attackSparkParticles.Emitting = false;
	}

	private void OnDebuffStart()
	{
		_debuffPuffParticles.Emitting = true;
	}

	private void OnDebuffEnd()
	{
		_debuffPuffParticles.Emitting = false;
	}

	private void OnDissipate()
	{
		Tween tween = CreateTween();
		tween.TweenMethod(Callable.From<float>(DissipateFunction), 0f, 1f, 1.399999976158142);
		tween.SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Quart);
	}

	private void DissipateFunction(float t)
	{
		for (int i = 0; i < _smokeMaterials.Count; i++)
		{
			_smokeMaterials[i].SetShaderParameter(_alphaStep, _smokeSteps[i] + (Vector2.One - _smokeSteps[i]) * t);
		}
	}

	private void OnDeathBreathStart()
	{
		_gasPuffParticles.Emitting = true;
	}

	private void OnDeathBreathEnd()
	{
		_gasPuffParticles.Emitting = false;
	}

	private void OnReconstitute()
	{
		for (int i = 0; i < _smokeMaterials.Count; i++)
		{
			_smokeMaterials[i].SetShaderParameter(_alphaStep, _smokeSteps[i]);
		}
	}
}
