using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Models.Monsters;
using MegaCrit.Sts2.Core.Models.Powers;
using MegaCrit.Sts2.Core.Nodes.Audio;

namespace MegaCrit.Sts2.Core.Nodes.Vfx.Backgrounds;

[GlobalClass]
public partial class NCeremonialBeastBgVfx : Node
{
	private bool _isGlowOn;

	private bool _areSkullsOn;

	private Node2D _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_parent.Visible = false;
		_animController = new MegaSprite(_parent);
	}

	public override void _EnterTree()
	{
		CombatManager.Instance.StateTracker.CombatStateChanged += UpdateState;
	}

	public override void _ExitTree()
	{
		CombatManager.Instance.StateTracker.CombatStateChanged -= UpdateState;
	}

	private void UpdateState(CombatState combatState)
	{
		UpdateRingingSfx(combatState);
		UpdateVfxAndMusic(combatState);
	}

	private void UpdateRingingSfx(CombatState combatState)
	{
		bool flag = LocalContext.GetMe(combatState).Creature.HasPower<RingingPower>();
		NRunMusicController.Instance?.UpdateMusicParameter("ringing", flag ? 1 : 0);
	}

	private void UpdateVfxAndMusic(CombatState combatState)
	{
		Creature creature = combatState.Creatures.FirstOrDefault((Creature c) => c.Monster is CeremonialBeast);
		if (creature == null)
		{
			NRunMusicController.Instance?.UpdateMusicParameter("ceremonial_beast_progress", 5f);
			PlayFlowers();
			return;
		}
		if ((float)creature.CurrentHp > (float)creature.MaxHp * 0.66f)
		{
			_parent.Visible = false;
			return;
		}
		_parent.Visible = true;
		if ((float)creature.CurrentHp > (float)creature.MaxHp * 0.33f)
		{
			NRunMusicController.Instance?.UpdateMusicParameter("ceremonial_beast_progress", 1f);
			PlayGlow();
		}
		else if (creature.IsAlive)
		{
			NRunMusicController.Instance?.UpdateMusicParameter("ceremonial_beast_progress", 1f);
			PlaySkulls();
		}
		else
		{
			NRunMusicController.Instance?.UpdateMusicParameter("ceremonial_beast_progress", 5f);
			PlayFlowers();
		}
	}

	private void PlayGlow()
	{
		if (!_isGlowOn)
		{
			_isGlowOn = true;
			MegaAnimationState animationState = _animController.GetAnimationState();
			animationState.SetAnimation("glow_spawn");
			animationState.AddAnimation("glow_idle");
		}
	}

	private void PlaySkulls()
	{
		if (!_areSkullsOn)
		{
			_areSkullsOn = true;
			MegaAnimationState animationState = _animController.GetAnimationState();
			animationState.SetAnimation("skulls_spawn");
			animationState.AddAnimation("glow_and_skulls_idle");
		}
	}

	private void PlayFlowers()
	{
		MegaAnimationState animationState = _animController.GetAnimationState();
		animationState.SetAnimation("glow_and_skulls_idle");
		animationState.AddAnimation("plants_spawn", 4.5f, loop: false);
	}
}
