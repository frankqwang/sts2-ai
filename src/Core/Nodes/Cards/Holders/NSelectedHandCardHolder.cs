using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.Nodes.Cards.Holders;

public partial class NSelectedHandCardHolder : NCardHolder
{
	private Tween? _tween;

	private static string ScenePath => SceneHelper.GetScenePath("/cards/holders/selected_hand_card_holder");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	public static NSelectedHandCardHolder Create(NHandCardHolder originalHolder)
	{
		NCard cardNode = originalHolder.CardNode;
		NSelectedHandCardHolder nSelectedHandCardHolder = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NSelectedHandCardHolder>(PackedScene.GenEditState.Disabled);
		originalHolder.Clear();
		nSelectedHandCardHolder.Name = $"SelectedHandCardHolder-{cardNode.Model.Id}";
		nSelectedHandCardHolder.SetCard(cardNode);
		nSelectedHandCardHolder.Scale = nSelectedHandCardHolder.SmallScale;
		return nSelectedHandCardHolder;
	}

	public override void _Ready()
	{
		ConnectSignals();
		_tween = CreateTween();
		_tween.TweenProperty(base.CardNode, "position", base.Position, 0.15000000596046448).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		_tween.Play();
	}

	protected override void CreateHoverTips()
	{
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
	}
}
