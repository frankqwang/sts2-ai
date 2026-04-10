using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

[GlobalClass]
public partial class NSoulFyshVfx : Node
{
	private static readonly StringName _amount = new StringName("amount");

	private ShaderMaterial? _soundShaderMat;

	private MegaSlotNode _soundSlotNode;

	private ShaderMaterial? _beckonShaderMat;

	private MegaSlotNode _beckonSlotNode;

	private Node2D _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_animController.ConnectAnimationEvent(Callable.From<GodotObject, GodotObject, GodotObject, GodotObject>(OnAnimationEvent));
		_soundSlotNode = new MegaSlotNode(_parent.GetNode("Soundwave"));
		_soundShaderMat = _soundSlotNode.GetNormalMaterial() as ShaderMaterial;
		_soundShaderMat?.SetShaderParameter(_amount, 0.3f);
		_beckonSlotNode = new MegaSlotNode(_parent.GetNode("Beckonwave"));
		_beckonShaderMat = _beckonSlotNode.GetNormalMaterial() as ShaderMaterial;
		_beckonShaderMat?.SetShaderParameter(_amount, 0.3f);
		_animController.GetAnimationState().SetAnimation("attack_debuff");
	}

	private void OnAnimationEvent(GodotObject _, GodotObject __, GodotObject ___, GodotObject spineEvent)
	{
		switch (new MegaEvent(spineEvent).GetData().GetEventName())
		{
		case "soundwave_start":
			StartSoundwave();
			break;
		case "soundwave_end":
			EndSoundwave();
			break;
		case "beckon_start":
			StartBeckon();
			break;
		case "beckon_end":
			EndBeckon();
			break;
		}
	}

	private void StartSoundwave()
	{
		Tween tween = CreateTween();
		tween.SetEase(Tween.EaseType.Out);
		tween.SetTrans(Tween.TransitionType.Quad);
		tween.TweenProperty(_soundShaderMat, "shader_parameter/amount", 1f, 0.44999998807907104);
	}

	private void EndSoundwave()
	{
		Tween tween = CreateTween();
		tween.SetEase(Tween.EaseType.In);
		tween.SetTrans(Tween.TransitionType.Quad);
		tween.TweenProperty(_soundShaderMat, "shader_parameter/amount", 0.3f, 0.5);
	}

	private void StartBeckon()
	{
		Tween tween = CreateTween();
		tween.SetEase(Tween.EaseType.Out);
		tween.SetTrans(Tween.TransitionType.Quad);
		tween.TweenProperty(_beckonShaderMat, "shader_parameter/amount", 1f, 0.25);
	}

	private void EndBeckon()
	{
		Tween tween = CreateTween();
		tween.SetEase(Tween.EaseType.In);
		tween.SetTrans(Tween.TransitionType.Quad);
		tween.TweenProperty(_beckonShaderMat, "shader_parameter/amount", 0.3f, 0.5);
	}
}
