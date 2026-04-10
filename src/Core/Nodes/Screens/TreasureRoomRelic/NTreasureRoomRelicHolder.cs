using Godot;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Relics;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Relics;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;

public partial class NTreasureRoomRelicHolder : NButton
{
	private GpuParticles2D _uncommonGlow;

	private GpuParticles2D _rareGlow;

	private bool _animatedIn;

	private Tween? _tween;

	private Tween? _initTween;

	public int Index { get; set; }

	public NMultiplayerVoteContainer VoteContainer { get; private set; }

	public NRelic Relic { get; private set; }

	public override void _Ready()
	{
		ConnectSignals();
		VoteContainer = GetNode<NMultiplayerVoteContainer>("%MultiplayerVoteContainer");
		Relic = GetNode<NRelic>("%Relic");
		_uncommonGlow = GetNode<GpuParticles2D>("%UncommonGlow");
		_rareGlow = GetNode<GpuParticles2D>("%RareGlow");
	}

	public void Initialize(RelicModel relic, IRunState runState)
	{
		Relic.Model = relic;
		VoteContainer.Initialize(PlayerVotedForRelic, runState.Players);
		Relic.Modulate = StsColors.transparentBlack;
		_initTween?.Kill();
		_initTween = CreateTween().SetParallel();
		_initTween.TweenProperty(Relic, "modulate", Colors.White, 0.25);
		if (Relic.Model.Rarity == RelicRarity.Uncommon)
		{
			Tween tween = CreateTween().SetParallel();
			_uncommonGlow.Visible = true;
			_uncommonGlow.Modulate = StsColors.transparentWhite;
			_uncommonGlow.GlobalPosition = Relic.GlobalPosition + Vector2.One * 68f;
			tween.TweenProperty(_uncommonGlow, "modulate", Colors.White, 0.25);
		}
		else if (Relic.Model.Rarity == RelicRarity.Rare)
		{
			Tween tween2 = CreateTween().SetParallel();
			_rareGlow.Visible = true;
			_rareGlow.Modulate = StsColors.transparentWhite;
			_rareGlow.GlobalPosition = Relic.GlobalPosition + Vector2.One * 68f;
			tween2.TweenProperty(_rareGlow, "modulate", Colors.White, 0.25);
		}
	}

	private bool PlayerVotedForRelic(Player player)
	{
		return RunManager.Instance.TreasureRoomRelicSynchronizer.GetPlayerVote(player) == Index;
	}

	public void AnimateAwayVotes()
	{
		CreateTween().TweenProperty(VoteContainer, "modulate:a", 0f, 0.25);
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(Relic, "scale", Vector2.One * 2.1f, 0.05);
		NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, Relic.Model.HoverTips);
		nHoverTipSet.SetAlignmentForRelic(Relic);
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(Relic, "scale", Vector2.One * 2f, 0.4).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		NHoverTipSet.Remove(this);
	}

	protected override void OnPress()
	{
		base.OnPress();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(Relic, "scale", Vector2.One * 1.9f, 0.4).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	protected override void OnRelease()
	{
		base.OnRelease();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(Relic, "scale", Vector2.One * 2f, 0.05).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}
}
