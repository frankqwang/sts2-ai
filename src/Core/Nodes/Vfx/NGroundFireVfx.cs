using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NGroundFireVfx : Node2D
{
	private static readonly StringName _outerColor = new StringName("OuterColor");

	private static readonly StringName _innerColor = new StringName("InnerColor");

	private Tween? _tween;

	private static readonly string _scenePath = SceneHelper.GetScenePath("vfx/fires/vfx_ground_fire");

	private Node2D _mainFire;

	private GpuParticles2D _ember;

	private GpuParticles2D _flameSprites;

	private VfxColor _vfxColor;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public static NGroundFireVfx? Create(Creature target, VfxColor color = VfxColor.Red)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCreature creatureNode = NCombatRoom.Instance.GetCreatureNode(target);
		if (creatureNode == null)
		{
			return null;
		}
		NGroundFireVfx nGroundFireVfx = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NGroundFireVfx>(PackedScene.GenEditState.Disabled);
		nGroundFireVfx._vfxColor = color;
		nGroundFireVfx.GlobalPosition = creatureNode.GetBottomOfHitbox();
		return nGroundFireVfx;
	}

	public override void _Ready()
	{
		_mainFire = GetNode<Node2D>("MainFire");
		_ember = GetNode<GpuParticles2D>("Ember");
		_flameSprites = GetNode<GpuParticles2D>("FlameSprites");
		ApplyColor();
		TaskHelper.RunSafely(AnimateIn());
	}

	private void ApplyColor()
	{
		if (_vfxColor != VfxColor.Red)
		{
			Color color = Colors.White;
			Color color2 = Colors.White;
			Color color3 = new Color("541b00");
			switch (_vfxColor)
			{
			case VfxColor.Green:
				color = new Color("2fa800");
				color2 = new Color("06a000");
				color3 = new Color("541b00");
				break;
			case VfxColor.Blue:
				color = new Color("0099cd");
				color2 = new Color("00a3bf");
				color3 = Colors.Black;
				break;
			case VfxColor.Purple:
				color = new Color("7821ff");
				color2 = new Color("3f21ff");
				color3 = new Color("541b00");
				break;
			default:
				throw new ArgumentOutOfRangeException();
			case VfxColor.Black:
			case VfxColor.White:
				break;
			}
			Node node = _mainFire.GetNode("VfxAdditiveStepFire");
			ShaderMaterial shaderMaterial = (ShaderMaterial)node.GetNode<Node2D>("SteppedFireMix").Material;
			shaderMaterial.SetShaderParameter(_outerColor, color);
			shaderMaterial.SetShaderParameter(_innerColor, color2);
			shaderMaterial = (ShaderMaterial)node.GetNode<Node2D>("SteppedFireAdd").Material;
			shaderMaterial.SetShaderParameter(_outerColor, color3);
		}
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
	}

	private async Task AnimateIn()
	{
		_mainFire.Modulate = Colors.Transparent;
		_mainFire.Scale = Vector2.Zero;
		_ember.Emitting = true;
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_mainFire, "scale", Vector2.One * 4f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back);
		_tween.TweenProperty(_mainFire, "modulate:a", 0.9f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		await ToSignal(_tween, Tween.SignalName.Finished);
		_flameSprites.Emitting = true;
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_mainFire, "modulate:a", 0f, 0.5);
		_tween.TweenProperty(_flameSprites, "modulate:a", 0f, 0.5);
		_tween.TweenProperty(_mainFire, "scale", Vector2.One * 2f, 2.0).SetEase(Tween.EaseType.InOut).SetTrans(Tween.TransitionType.Cubic);
		await ToSignal(_tween, Tween.SignalName.Finished);
		_flameSprites.Emitting = false;
		await ToSignal(_ember, GpuParticles2D.SignalName.Finished);
		this.QueueFreeSafely();
	}
}
