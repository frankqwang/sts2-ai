using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Rewards;

public partial class NRewardButton : NButton
{
	[Signal]
	public delegate void RewardClaimedEventHandler(NRewardButton button);

	[Signal]
	public delegate void RewardSkippedEventHandler(NRewardButton button);

	private static readonly StringName _v = new StringName("v");

	private static readonly StringName _fontOutlineColor = new StringName("theme_override_colors/font_outline_color");

	private static readonly StringName _defaultColor = new StringName("theme_override_colors/default_color");

	private TextureRect _background;

	private Control _iconContainer;

	private MegaRichTextLabel _label;

	private NSelectionReticle _reticle;

	private ShaderMaterial _hsv;

	private Tween? _currentTween;

	private Variant _hsvDefault = 0.9;

	private Variant _hsvHover = 1.1;

	private Variant _hsvDown = 0.7;

	public Reward? Reward { get; private set; }

	private static string ScenePath => "res://scenes/rewards/reward_button.tscn";

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	public static NRewardButton Create(Reward reward, NRewardsScreen screen)
	{
		NRewardButton nRewardButton = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NRewardButton>(PackedScene.GenEditState.Disabled);
		nRewardButton.SetReward(reward);
		return nRewardButton;
	}

	public override void _Ready()
	{
		ConnectSignals();
		_background = GetNode<TextureRect>("%Background");
		_iconContainer = GetNode<Control>("%Icon");
		_label = GetNode<MegaRichTextLabel>("%Label");
		_reticle = GetNode<NSelectionReticle>("%SelectionReticle");
		_hsv = (ShaderMaterial)_background.Material;
		Reload();
	}

	private void SetReward(Reward reward)
	{
		if (reward is LinkedRewardSet)
		{
			throw new ArgumentException("You aren't allowed to apply a RewardChainSet to a NRewardButton");
		}
		Reward = reward;
		if (IsNodeReady())
		{
			Reload();
		}
	}

	private void Reload()
	{
		if (IsNodeReady() && Reward != null)
		{
			Control control = Reward.CreateIcon();
			_iconContainer.AddChildSafely(control);
			control.Position = Reward.IconPosition;
			if (Reward is PotionReward)
			{
				control.Scale = 0.8f * Vector2.One;
			}
			_label.Text = Reward.Description.GetFormattedText();
		}
	}

	private async Task GetReward()
	{
		Disable();
		if (await Reward.OnSelectWrapper())
		{
			if (TestMode.IsOff)
			{
				NGlobalUi globalUi = NRun.Instance.GlobalUi;
				Reward reward = Reward;
				if (reward is RelicReward relicReward)
				{
					RelicModel claimedRelic = relicReward.ClaimedRelic;
					if (claimedRelic != null)
					{
						globalUi.RelicInventory.AnimateRelic(relicReward.ClaimedRelic, _iconContainer.GlobalPosition);
					}
				}
				else if (reward is PotionReward potionReward)
				{
					globalUi.TopBar.PotionContainer.AnimatePotion(potionReward.ClaimedPotion, _iconContainer.GlobalPosition);
				}
			}
			_isEnabled = false;
			EmitSignal(SignalName.RewardClaimed, this);
		}
		else
		{
			Enable();
			this.TryGrabFocus();
			EmitSignal(SignalName.RewardSkipped, this);
		}
	}

	protected override void OnRelease()
	{
		base.OnRelease();
		OnUnfocus();
		TaskHelper.RunSafely(GetReward());
	}

	protected override void OnPress()
	{
		base.OnPress();
		_currentTween?.Kill();
		_currentTween = CreateTween().SetParallel();
		_currentTween.TweenMethod(Callable.From<float>(UpdateShaderParam), _hsvHover, _hsvDown, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_currentTween.TweenProperty(_label, "modulate:a", 1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		NHoverTipSet.Remove(this);
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_label.Set(_defaultColor, StsColors.gold);
		_label.Set(_fontOutlineColor, StsColors.rewardLabelGoldOutline);
		_currentTween?.Kill();
		_hsv.SetShaderParameter(_v, _hsvHover);
		NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, Reward.HoverTips);
		nHoverTipSet.GlobalPosition = base.GlobalPosition + Vector2.Left * 45f;
		nHoverTipSet.SetAlignment(this, HoverTipAlignment.Left);
		_reticle.OnSelect();
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		_label.Set(_defaultColor, StsColors.cream);
		_label.Set(_fontOutlineColor, StsColors.rewardLabelOutline);
		_currentTween?.Kill();
		_currentTween = CreateTween();
		_currentTween.TweenMethod(Callable.From<float>(UpdateShaderParam), _hsv.GetShaderParameter(_v), _hsvDefault, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		NHoverTipSet.Remove(this);
		_reticle.OnDeselect();
	}

	private void UpdateShaderParam(float value)
	{
		_hsv.SetShaderParameter(_v, value);
	}
}
