using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Audio;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Ascension;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models.Powers;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine;
using MegaCrit.Sts2.Core.Nodes.Audio;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Vfx;

namespace MegaCrit.Sts2.Core.Models.Monsters;

public sealed class Doormaker : MonsterModel
{
	private const string _doormakerTrackName = "queen_progress";

	private const string _faceVisual = "monsters/beta/door_maker_placeholder_2.png";

	private const string _teethVisual = "monsters/beta/door_maker_placeholder_3.png";

	private const string _handsVisual = "monsters/beta/door_maker_placeholder_4.png";

	private const string _dramaticOpenLine = "DOORMAKER.moves.DRAMATIC_OPEN.speakLine";

	private const int _graspHitCount = 2;

	private int _originalHp;

	private bool _isPortalOpen;

	public override LocString Title
	{
		get
		{
			if (!IsPortalOpen)
			{
				return MonsterModel.L10NMonsterLookup("DOOR.name");
			}
			return MonsterModel.L10NMonsterLookup(base.Id.Entry + ".name");
		}
	}

	public override int MinInitialHp => AscensionHelper.GetValueIfAscension(AscensionLevel.ToughEnemies, 512, 489);

	public override int MaxInitialHp => MinInitialHp;

	private int HungerDamage => AscensionHelper.GetValueIfAscension(AscensionLevel.DeadlyEnemies, 35, 30);

	private int ScrutinyDamage => AscensionHelper.GetValueIfAscension(AscensionLevel.DeadlyEnemies, 26, 24);

	private int GraspDamage => AscensionHelper.GetValueIfAscension(AscensionLevel.DeadlyEnemies, 11, 10);

	private int GraspStrengthGain => AscensionHelper.GetValueIfAscension(AscensionLevel.DeadlyEnemies, 4, 3);

	public override DamageSfxType TakeDamageSfxType => DamageSfxType.Armor;

	private int OriginalHp
	{
		get
		{
			return _originalHp;
		}
		set
		{
			AssertMutable();
			_originalHp = value;
		}
	}

	private bool IsPortalOpen
	{
		get
		{
			return _isPortalOpen;
		}
		set
		{
			AssertMutable();
			_isPortalOpen = value;
		}
	}

	public override async Task AfterAddedToRoom()
	{
		await base.AfterAddedToRoom();
		OriginalHp = base.Creature.MaxHp;
		await CreatureCmd.SetMaxAndCurrentHp(base.Creature, 999999999m);
		base.Creature.ShowsInfiniteHp = true;
	}

	protected override MonsterMoveStateMachine GenerateMoveStateMachine()
	{
		List<MonsterState> list = new List<MonsterState>();
		MoveState moveState = new MoveState("DRAMATIC_OPEN_MOVE", DramaticOpenMove, new SummonIntent());
		MoveState moveState2 = new MoveState("HUNGER_MOVE", HungerMove, new SingleAttackIntent(HungerDamage));
		MoveState moveState3 = new MoveState("SCRUTINY_MOVE", ScrutinyMove, new SingleAttackIntent(ScrutinyDamage));
		MoveState moveState4 = new MoveState("GRASP_MOVE", GraspMove, new MultiAttackIntent(GraspDamage, 2), new BuffIntent());
		moveState.FollowUpState = moveState2;
		moveState2.FollowUpState = moveState3;
		moveState3.FollowUpState = moveState4;
		moveState4.FollowUpState = moveState2;
		list.Add(moveState);
		list.Add(moveState2);
		list.Add(moveState3);
		list.Add(moveState4);
		return new MonsterMoveStateMachine(list, moveState);
	}

	private async Task DramaticOpenMove(IReadOnlyList<Creature> targets)
	{
		IsPortalOpen = true;
		await CreatureCmd.SetMaxAndCurrentHp(base.Creature, OriginalHp);
		List<PowerModel> list = base.Creature.Powers.ToList();
		foreach (PowerModel item in list)
		{
			await PowerCmd.Remove(item);
		}
		base.Creature.ShowsInfiniteHp = false;
		await SwapPhasePower<HungerPower>();
		UpdateVisual("monsters/beta/door_maker_placeholder_3.png");
		await Cmd.CustomScaledWait(0.2f, 0.6f);
		TalkCmd.Play(MonsterModel.L10NMonsterLookup("DOORMAKER.moves.DRAMATIC_OPEN.speakLine"), base.Creature, VfxColor.Purple);
		await Cmd.CustomScaledWait(0.2f, 0.6f);
		NRunMusicController.Instance?.UpdateMusicParameter("queen_progress", 1f);
	}

	private async Task HungerMove(IReadOnlyList<Creature> targets)
	{
		await DamageCmd.Attack(HungerDamage).FromMonster(this).WithAttackerAnim("Attack", 0.15f)
			.WithHitFx("vfx/vfx_attack_blunt")
			.Execute(null);
		await SwapPhasePower<ScrutinyPower>();
		await Cmd.Wait(0.2f);
		UpdateVisual("monsters/beta/door_maker_placeholder_2.png");
	}

	private async Task ScrutinyMove(IReadOnlyList<Creature> targets)
	{
		await DamageCmd.Attack(ScrutinyDamage).FromMonster(this).WithAttackerAnim("Attack", 0.15f)
			.WithHitFx("vfx/vfx_bite")
			.Execute(null);
		await SwapPhasePower<GraspPower>();
		await Cmd.Wait(0.2f);
		UpdateVisual("monsters/beta/door_maker_placeholder_4.png");
	}

	private async Task GraspMove(IReadOnlyList<Creature> targets)
	{
		await DamageCmd.Attack(GraspDamage).WithHitCount(2).FromMonster(this)
			.WithAttackerAnim("Attack", 0.15f)
			.WithHitFx("vfx/vfx_attack_blunt")
			.Execute(null);
		await PowerCmd.Apply<StrengthPower>(base.Creature, GraspStrengthGain, base.Creature, null);
		await SwapPhasePower<HungerPower>();
		await Cmd.Wait(0.2f);
		UpdateVisual("monsters/beta/door_maker_placeholder_3.png");
	}

	private async Task SwapPhasePower<T>() where T : PowerModel
	{
		if (base.Creature.HasPower<HungerPower>())
		{
			await PowerCmd.Remove<HungerPower>(base.Creature);
		}
		if (base.Creature.HasPower<ScrutinyPower>())
		{
			await PowerCmd.Remove<ScrutinyPower>(base.Creature);
		}
		if (base.Creature.HasPower<GraspPower>())
		{
			await PowerCmd.Remove<GraspPower>(base.Creature);
		}
		await PowerCmd.Apply<T>(base.Creature, 1m, base.Creature, null);
	}

	private void UpdateVisual(string path)
	{
		NCreature? nCreature = HeadlessVisualCompat.TryGetCreatureNode(base.Creature);
		Sprite2D? body = nCreature?.Visuals?.GetCurrentBody() as Sprite2D;
		if (nCreature == null || body == null)
		{
			return;
		}

		body.Texture = PreloadManager.Cache.GetTexture2D(ImageHelper.GetImagePath(path));
		Vector2 scale = body.Scale;
		Tween tween = nCreature.CreateTween();
		tween.TweenProperty(body, "scale", scale, 1.2000000476837158).From(scale * 0.5f).SetEase(Tween.EaseType.Out)
			.SetTrans(Tween.TransitionType.Sine);
		tween.Parallel().TweenProperty(body, "modulate", Colors.White, 0.5).From(Colors.Black);
	}

	public override Task AfterDeath(PlayerChoiceContext choiceContext, Creature creature, bool wasRemovalPrevented, float deathAnimLength)
	{
		if (creature != base.Creature)
		{
			return Task.CompletedTask;
		}
		NRunMusicController.Instance?.UpdateMusicParameter("queen_progress", 5f);
		return Task.CompletedTask;
	}
}
