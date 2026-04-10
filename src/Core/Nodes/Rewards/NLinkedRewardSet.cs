using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Rewards;

namespace MegaCrit.Sts2.Core.Nodes.Rewards;

public partial class NLinkedRewardSet : Control
{
	[Signal]
	public delegate void RewardClaimedEventHandler(NLinkedRewardSet linkedRewardSet);

	private NRewardsScreen _rewardsScreen;

	private Control _rewardContainer;

	private Control _chainsContainer;

	public LinkedRewardSet LinkedRewardSet { get; private set; }

	private static string ScenePath => SceneHelper.GetScenePath("/rewards/linked_reward_set");

	private static string ChainImagePath => ImageHelper.GetImagePath("/ui/reward_screen/reward_chain.png");

	public static IEnumerable<string> AssetPaths => new string[2] { ScenePath, ChainImagePath };

	public override void _Ready()
	{
		_rewardContainer = GetNode<Control>("%RewardContainer");
		_chainsContainer = GetNode<Control>("%ChainContainer");
		Reload();
	}

	public static NLinkedRewardSet Create(LinkedRewardSet linkedReward, NRewardsScreen screen)
	{
		NLinkedRewardSet nLinkedRewardSet = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NLinkedRewardSet>(PackedScene.GenEditState.Disabled);
		nLinkedRewardSet._rewardsScreen = screen;
		nLinkedRewardSet.SetReward(linkedReward);
		return nLinkedRewardSet;
	}

	private void SetReward(LinkedRewardSet linkedReward)
	{
		LinkedRewardSet = linkedReward;
		if (IsNodeReady())
		{
			Reload();
		}
	}

	private void Reload()
	{
		if (!IsNodeReady())
		{
			return;
		}
		for (int i = 0; i < LinkedRewardSet.Rewards.Count; i++)
		{
			Reward reward = LinkedRewardSet.Rewards[i];
			NRewardButton nRewardButton = NRewardButton.Create(reward, _rewardsScreen);
			nRewardButton.CustomMinimumSize -= Vector2.Right * 20f;
			_rewardContainer.AddChildSafely(nRewardButton);
			nRewardButton.Connect(NRewardButton.SignalName.RewardClaimed, Callable.From(GetReward));
			if (i < LinkedRewardSet.Rewards.Count - 1)
			{
				TextureRect textureRect = new TextureRect();
				textureRect.MouseFilter = MouseFilterEnum.Ignore;
				textureRect.Texture = PreloadManager.Cache.GetCompressedTexture2D(ChainImagePath);
				textureRect.Size = Vector2.One * 50f;
				_chainsContainer.AddChildSafely(textureRect);
				textureRect.GlobalPosition = _chainsContainer.GlobalPosition + Vector2.Down * i * (3f + nRewardButton.CustomMinimumSize.Y);
			}
		}
	}

	private void GetReward()
	{
		_rewardsScreen.RewardCollectedFrom(this);
		LinkedRewardSet.OnSkipped();
		EmitSignal(SignalName.RewardClaimed);
		this.QueueFreeSafely();
	}
}
