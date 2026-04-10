using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Cards;

public partial class NCardBundle : Control
{
	[Signal]
	public delegate void ClickedEventHandler(NCardBundle cardHolder);

	private const float _cardSeparation = 45f;

	private readonly Vector2 _hoverScale = Vector2.One * 0.85f;

	public readonly Vector2 smallScale = Vector2.One * 0.8f;

	private Control _cardHolder;

	private readonly List<NCard> _cardNodes = new List<NCard>();

	private Tween? _hoverTween;

	private Tween? _cardTween;

	public NClickableControl Hitbox { get; private set; }

	public IReadOnlyList<CardModel> Bundle { get; private set; }

	public IReadOnlyList<NCard> CardNodes => _cardNodes;

	private static string ScenePath => SceneHelper.GetScenePath("/cards/card_bundle");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	public static NCardBundle? Create(IReadOnlyList<CardModel> bundle)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCardBundle nCardBundle = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NCardBundle>(PackedScene.GenEditState.Disabled);
		nCardBundle.Name = "NCardBundle";
		nCardBundle.Scale = nCardBundle.smallScale;
		nCardBundle.Bundle = bundle;
		return nCardBundle;
	}

	public override void _Ready()
	{
		Hitbox = GetNode<NButton>("%Hitbox");
		_cardHolder = GetNode<Control>("%Cards");
		Hitbox.Connect(NClickableControl.SignalName.Focused, Callable.From<NClickableControl>(OnFocused));
		Hitbox.Connect(NClickableControl.SignalName.Unfocused, Callable.From<NClickableControl>(OnUnfocused));
		Hitbox.Connect(NClickableControl.SignalName.Released, Callable.From<NClickableControl>(OnClicked));
		for (int i = 0; i < Bundle.Count; i++)
		{
			NCard nCard = NCard.Create(Bundle[i]);
			_cardHolder = GetNode<Control>("%Cards");
			_cardHolder.AddChildSafely(nCard);
			nCard.UpdateVisuals(PileType.None, CardPreviewMode.Normal);
			nCard.Position += new Vector2(-1f, 1f) * 45f * ((float)i - (float)Bundle.Count / 2f);
			float num = 0.5f + (float)i / (float)(Bundle.Count - 1) * 0.5f;
			nCard.Modulate = new Color(num, num, num);
			_cardNodes.Add(nCard);
		}
	}

	public IReadOnlyList<NCard> RemoveCardNodes()
	{
		_cardTween?.Kill();
		_cardTween = CreateTween().SetParallel();
		foreach (NCard cardNode in _cardNodes)
		{
			_cardTween.TweenProperty(cardNode, "modulate", Colors.White, 0.15000000596046448);
		}
		return CardNodes;
	}

	public void ReAddCardNodes()
	{
		_cardTween?.Kill();
		_cardTween = CreateTween().SetParallel();
		for (int i = 0; i < _cardNodes.Count; i++)
		{
			NCard nCard = _cardNodes[i];
			Vector2 globalPosition = nCard.GlobalPosition;
			nCard.GetParent()?.RemoveChildSafely(nCard);
			_cardHolder = GetNode<Control>("%Cards");
			_cardHolder.AddChildSafely(nCard);
			nCard.GlobalPosition = globalPosition;
			nCard.UpdateVisuals(PileType.None, CardPreviewMode.Normal);
			_cardTween.TweenProperty(nCard, "position", new Vector2(-1f, 1f) * 45f * ((float)i - (float)_cardNodes.Count / 2f), 0.4000000059604645).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			float num = 0.5f + (float)i / (float)(_cardNodes.Count - 1) * 0.5f;
			_cardTween.TweenProperty(nCard, "modulate", new Color(num, num, num), 0.4000000059604645);
		}
	}

	private void OnClicked(NClickableControl _)
	{
		EmitSignal(SignalName.Clicked, this);
	}

	private void OnFocused(NClickableControl _)
	{
		_hoverTween?.Kill();
		base.Scale = _hoverScale;
	}

	private void OnUnfocused(NClickableControl _)
	{
		_hoverTween?.Kill();
		_hoverTween = CreateTween();
		_hoverTween.TweenProperty(this, "scale", smallScale, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	public override void _ExitTree()
	{
		foreach (NCard cardNode in _cardNodes)
		{
			if (IsAncestorOf(cardNode))
			{
				cardNode.QueueFreeSafely();
			}
		}
	}
}
