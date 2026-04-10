using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;
using MegaCrit.Sts2.Core.Commands;

namespace MegaCrit.Sts2.Core.Nodes.Vfx.Backgrounds;

[GlobalClass]
public partial class NKaiserCrabBossBackground : Node
{
	private enum RightArmState
	{
		Default,
		Charging,
		Resting
	}

	public enum ArmSide
	{
		Left,
		Right
	}

	private const int _bodyTrack = 0;

	private const int _leftArmTrack = 1;

	private const int _rightArmTrack = 2;

	private const int _reactionTrack = 3;

	private MegaSprite _animController;

	private Node2D _leftArm;

	private Node2D _rightArm;

	private RightArmState _rightArmState;

	public override void _Ready()
	{
		_animController = new MegaSprite(GetNode<Node2D>("%Visuals"));
		_leftArm = GetNode<Node2D>("%ArmBoneL");
		_rightArm = GetNode<Node2D>("%ArmBoneR");
		MegaAnimationState animationState = _animController.GetAnimationState();
		animationState.SetAnimation("right/idle_loop", loop: true, 2);
		animationState.SetAnimation("body/idle_loop");
		animationState.SetAnimation("left/idle_loop", loop: true, 1);
	}

	public async Task PlayAttackAnim(ArmSide side, string animation, float duration)
	{
		string text = side.ToString().ToLowerInvariant();
		MegaAnimationState animationState = _animController.GetAnimationState();
		animationState.SetAnimation(text + "/" + animation, loop: false, (side == ArmSide.Left) ? 1 : 2);
		animationState.SetAnimation("reactions/attack_" + text, loop: false, 3);
		AddEmptyReactionAnimation(animationState);
		animationState.AddAnimation(text + "/idle_loop", 0f, loop: true, (side == ArmSide.Left) ? 1 : 2);
		await Cmd.Wait(duration);
	}

	public void PlayHurtAnim(ArmSide side)
	{
		string text = side.ToString().ToLowerInvariant();
		MegaAnimationState animationState = _animController.GetAnimationState();
		animationState.SetAnimation("reactions/hurt_" + text, loop: false, 3);
		AddEmptyReactionAnimation(animationState);
		if (side == ArmSide.Left)
		{
			animationState.SetAnimation(text + "/hurt", loop: false, 1);
			animationState.AddAnimation(text + "/idle_loop", 0f, loop: true, 1);
			return;
		}
		switch (_rightArmState)
		{
		case RightArmState.Default:
			animationState.SetAnimation(text + "/hurt", loop: false, 2);
			animationState.AddAnimation("right/idle_loop", 0f, loop: true, 2);
			break;
		case RightArmState.Charging:
			animationState.SetAnimation(text + "/hurt_charged", loop: false, 2);
			animationState.AddAnimation("right/charged_loop", 0f, loop: true, 2);
			break;
		case RightArmState.Resting:
			animationState.SetAnimation(text + "/hurt_resting", loop: false, 2);
			animationState.AddAnimation("right/rest_loop", 0f, loop: true, 2);
			break;
		}
	}

	public void PlayArmDeathAnim(ArmSide side)
	{
		string text = side.ToString().ToLowerInvariant();
		MegaAnimationState animationState = _animController.GetAnimationState();
		animationState.SetAnimation("reactions/hurt_" + text, loop: false, 3);
		AddEmptyReactionAnimation(animationState);
		if (side == ArmSide.Left)
		{
			animationState.SetAnimation(text + "/die", loop: false, 1);
			return;
		}
		switch (_rightArmState)
		{
		case RightArmState.Default:
		case RightArmState.Charging:
			animationState.SetAnimation("right/die", loop: false, 2);
			break;
		case RightArmState.Resting:
			animationState.SetAnimation("right/die_resting", loop: false, 2);
			break;
		}
	}

	public async Task PlayRightSideChargeUpAnim(float duration)
	{
		_rightArmState = RightArmState.Charging;
		MegaAnimationState animationState = _animController.GetAnimationState();
		animationState.SetAnimation("right/charge_up", loop: false, 2);
		animationState.AddAnimation("right/charged_loop", 0f, loop: true, 2);
		animationState.SetAnimation("reactions/attack_right", loop: false, 3);
		AddEmptyReactionAnimation(animationState);
		await Cmd.Wait(duration);
	}

	public async Task PlayRightSideHeavy(float duration)
	{
		_rightArmState = RightArmState.Resting;
		MegaAnimationState animationState = _animController.GetAnimationState();
		animationState.SetAnimation("right/attack_heavy", loop: false, 2);
		animationState.AddAnimation("right/rest_loop", 0f, loop: true, 2);
		animationState.SetAnimation("reactions/attack_right", loop: false, 3);
		AddEmptyReactionAnimation(animationState);
		await Cmd.Wait(duration);
	}

	public async Task PlayRightRecharge(float duration)
	{
		_rightArmState = RightArmState.Default;
		MegaAnimationState animationState = _animController.GetAnimationState();
		animationState.SetAnimation("right/wake_up", loop: false, 2);
		animationState.AddAnimation("right/idle_loop", 0f, loop: true, 2);
		await Cmd.Wait(duration);
	}

	public void PlayBodyDeathAnim()
	{
		MegaAnimationState animationState = _animController.GetAnimationState();
		animationState.SetAnimation("body/die", loop: false);
	}

	private void AddEmptyReactionAnimation(MegaAnimationState state)
	{
		state.AddEmptyAnimation(3);
	}
}
