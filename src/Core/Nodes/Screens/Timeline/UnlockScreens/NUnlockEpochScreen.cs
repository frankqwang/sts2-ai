using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Timeline;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Timeline.UnlockScreens;

public partial class NUnlockEpochScreen : NUnlockScreen
{
	private IReadOnlyList<EpochModel> _unlockedEpochs;

	private Tween? _cardFlyTween;

	private const double _initDelay = 0.3;

	private RichTextLabel _infoLabel;

	public override void _Ready()
	{
		ConnectSignals();
		_infoLabel = GetNode<RichTextLabel>("%InfoLabel");
		LocString locString = new LocString("timeline", "UNLOCK_EPOCHS");
		_infoLabel.Text = "[center]" + locString.GetFormattedText() + "[/center]";
		_infoLabel.Modulate = StsColors.transparentWhite;
	}

	public override void Open()
	{
		base.Open();
		_cardFlyTween = CreateTween().SetParallel();
		double num = 0.3;
		Vector2 position = GetNode<Control>("%Center").Position;
		PackedScene scene = PreloadManager.Cache.GetScene("res://scenes/timeline_screen/epoch.tscn");
		if (_unlockedEpochs.Count == 3)
		{
			for (int i = 0; i < _unlockedEpochs.Count; i++)
			{
				NEpochCard nEpochCard = scene.Instantiate<NEpochCard>(PackedScene.GenEditState.Disabled);
				nEpochCard.Init(_unlockedEpochs[i]);
				Control node = GetNode<Control>($"Slot{i}");
				node.AddChildSafely(nEpochCard);
				nEpochCard.SetToWigglyUnlockPreviewMode();
				_cardFlyTween.TweenProperty(node, "modulate", Colors.White, 1.0).SetDelay(num - 0.3);
				_cardFlyTween.TweenProperty(node, "position", node.Position, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back)
					.SetDelay(num);
				node.Modulate = StsColors.transparentBlack;
				node.Position = position;
				num += 0.25;
			}
		}
		else if (_unlockedEpochs.Count == 2)
		{
			for (int j = 0; j < _unlockedEpochs.Count; j++)
			{
				NEpochCard nEpochCard2 = scene.Instantiate<NEpochCard>(PackedScene.GenEditState.Disabled);
				nEpochCard2.Init(_unlockedEpochs[j]);
				Control node2 = GetNode<Control>($"Slot{3 + j}");
				node2.AddChildSafely(nEpochCard2);
				nEpochCard2.SetToWigglyUnlockPreviewMode();
				_cardFlyTween.TweenProperty(node2, "modulate", Colors.White, 1.0).SetDelay(num - 0.3);
				_cardFlyTween.TweenProperty(node2, "position", node2.Position, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back)
					.SetDelay(num);
				node2.Modulate = StsColors.transparentBlack;
				node2.Position = position;
				num += 0.33;
			}
		}
		else
		{
			Log.Error("Unlocking exactly 1 OR more than 3 Epochs are not supported.");
		}
		_cardFlyTween.TweenProperty(_infoLabel, "modulate", Colors.White, 1.0).SetDelay(0.25);
	}

	public void SetUnlocks(IReadOnlyList<EpochModel> epochs)
	{
		_unlockedEpochs = epochs;
	}
}
