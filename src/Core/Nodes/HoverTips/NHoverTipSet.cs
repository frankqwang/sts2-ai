using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Relics;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.HoverTips;

public partial class NHoverTipSet : Control
{
	public static bool shouldBlockHoverTips = false;

	private static readonly StringName _cardHoverTipContainerStr = new StringName("cardHoverTipContainer");

	private static readonly StringName _textHoverTipContainerStr = new StringName("textHoverTipContainer");

	private const float _hoverTipSpacing = 5f;

	private const float _hoverTipWidth = 360f;

	private const string _tipScenePath = "res://scenes/ui/hover_tip.tscn";

	private const string _tipSetScenePath = "res://scenes/ui/hover_tip_set.tscn";

	private const string _debuffMatPath = "res://materials/ui/hover_tip_debuff.tres";

	private static readonly Dictionary<Control, NHoverTipSet> _activeHoverTips = new Dictionary<Control, NHoverTipSet>();

	private VFlowContainer _textHoverTipContainer;

	private NHoverTipCardContainer _cardHoverTipContainer;

	private Control _owner;

	private bool _followOwner;

	private Vector2 _followOffset;

	private Vector2 _extraOffset = Vector2.Zero;

	private static Node HoverTipsContainer => NGame.Instance.HoverTipsContainer;

	private Vector2 TextHoverTipDimensions => _textHoverTipContainer.Size;

	private Vector2 CardHoverTipDimensions => _cardHoverTipContainer.Size;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlyArray<string>(new string[3] { "res://scenes/ui/hover_tip.tscn", "res://scenes/ui/hover_tip_set.tscn", "res://materials/ui/hover_tip_debuff.tres" });

	public void SetFollowOwner()
	{
		_followOwner = true;
		_followOffset = _owner.GlobalPosition - base.GlobalPosition;
	}

	public static NHoverTipSet CreateAndShow(Control owner, IHoverTip hoverTip, HoverTipAlignment alignment = HoverTipAlignment.None)
	{
		return CreateAndShow(owner, new global::_003C_003Ez__ReadOnlySingleElementList<IHoverTip>(hoverTip), alignment);
	}

	public static NHoverTipSet CreateAndShow(Control owner, IEnumerable<IHoverTip> hoverTips, HoverTipAlignment alignment = HoverTipAlignment.None)
	{
		NHoverTipSet nHoverTipSet = PreloadManager.Cache.GetScene("res://scenes/ui/hover_tip_set.tscn").Instantiate<NHoverTipSet>(PackedScene.GenEditState.Disabled);
		HoverTipsContainer.AddChildSafely(nHoverTipSet);
		if (shouldBlockHoverTips)
		{
			return nHoverTipSet;
		}
		_activeHoverTips.Add(owner, nHoverTipSet);
		nHoverTipSet.Init(owner, hoverTips);
		if (NGame.IsDebugHidingHoverTips)
		{
			nHoverTipSet.Visible = false;
		}
		owner.Connect(Node.SignalName.TreeExiting, Callable.From(delegate
		{
			Remove(owner);
		}));
		nHoverTipSet.SetAlignment(owner, alignment);
		return nHoverTipSet;
	}

	public static NHoverTipSet CreateAndShowMapPointHistory(Control owner, NMapPointHistoryHoverTip historyHoverTip)
	{
		NHoverTipSet nHoverTipSet = PreloadManager.Cache.GetScene("res://scenes/ui/hover_tip_set.tscn").Instantiate<NHoverTipSet>(PackedScene.GenEditState.Disabled);
		nHoverTipSet._owner = owner;
		HoverTipsContainer.AddChildSafely(nHoverTipSet);
		_activeHoverTips.Add(owner, nHoverTipSet);
		nHoverTipSet._textHoverTipContainer.AddChildSafely(historyHoverTip);
		if (NGame.IsDebugHidingHoverTips)
		{
			nHoverTipSet.Visible = false;
		}
		owner.Connect(Node.SignalName.TreeExiting, Callable.From(delegate
		{
			Remove(owner);
		}));
		return nHoverTipSet;
	}

	public override void _Ready()
	{
		_textHoverTipContainer = new VFlowContainer();
		_textHoverTipContainer.Name = _textHoverTipContainerStr;
		_textHoverTipContainer.MouseFilter = MouseFilterEnum.Ignore;
		this.AddChildSafely(_textHoverTipContainer);
		_cardHoverTipContainer = new NHoverTipCardContainer();
		_cardHoverTipContainer.Name = _cardHoverTipContainerStr;
		_cardHoverTipContainer.MouseFilter = MouseFilterEnum.Ignore;
		this.AddChildSafely(_cardHoverTipContainer);
	}

	public override void _Process(double delta)
	{
		if (_followOwner && _owner != null)
		{
			base.GlobalPosition = _owner.GlobalPosition - _followOffset + _extraOffset;
		}
	}

	private void Init(Control owner, IEnumerable<IHoverTip> hoverTips)
	{
		_owner = owner;
		foreach (IHoverTip item in IHoverTip.RemoveDupes(hoverTips))
		{
			if (item is HoverTip hoverTip)
			{
				Control control = PreloadManager.Cache.GetScene("res://scenes/ui/hover_tip.tscn").Instantiate<Control>(PackedScene.GenEditState.Disabled);
				_textHoverTipContainer.AddChildSafely(control);
				MegaLabel node = control.GetNode<MegaLabel>("%Title");
				if (hoverTip.Title == null)
				{
					node.Visible = false;
				}
				else
				{
					node.SetTextAutoSize(hoverTip.Title);
				}
				control.GetNode<MegaRichTextLabel>("%Description").Text = hoverTip.Description;
				control.GetNode<MegaRichTextLabel>("%Description").AutowrapMode = (TextServer.AutowrapMode)(hoverTip.ShouldOverrideTextOverflow ? 0 : 3);
				control.GetNode<TextureRect>("%Icon").Texture = hoverTip.Icon;
				if (hoverTip.IsDebuff)
				{
					control.GetNode<CanvasItem>("%Bg").Material = PreloadManager.Cache.GetMaterial("res://materials/ui/hover_tip_debuff.tres");
				}
				control.ResetSize();
				if (_textHoverTipContainer.Size.Y + control.Size.Y + 5f < NGame.Instance.GetViewportRect().Size.Y - 50f)
				{
					_textHoverTipContainer.Size = new Vector2(360f, _textHoverTipContainer.Size.Y + control.Size.Y + 5f);
				}
				else
				{
					_textHoverTipContainer.Alignment = FlowContainer.AlignmentMode.Center;
				}
			}
			else
			{
				_cardHoverTipContainer.Add((CardHoverTip)item);
			}
			AbstractModel canonicalModel = item.CanonicalModel;
			if (!(canonicalModel is CardModel card))
			{
				if (!(canonicalModel is RelicModel relic))
				{
					if (canonicalModel is PotionModel potion)
					{
						SaveManager.Instance.MarkPotionAsSeen(potion);
					}
				}
				else
				{
					SaveManager.Instance.MarkRelicAsSeen(relic);
				}
			}
			else
			{
				SaveManager.Instance.MarkCardAsSeen(card);
			}
		}
	}

	public void SetAlignment(Control node, HoverTipAlignment alignment)
	{
		if (alignment != HoverTipAlignment.None)
		{
			_textHoverTipContainer.Position = Vector2.Zero;
			switch (alignment)
			{
			case HoverTipAlignment.Left:
				_textHoverTipContainer.GlobalPosition = node.GlobalPosition;
				_textHoverTipContainer.Position += Vector2.Left * _textHoverTipContainer.Size.X;
				_textHoverTipContainer.ReverseFill = true;
				_cardHoverTipContainer.LayoutResizeAndReposition(node.GlobalPosition + new Vector2(node.Size.X, 0f) * node.Scale, HoverTipAlignment.Right);
				break;
			case HoverTipAlignment.Right:
				_cardHoverTipContainer.LayoutResizeAndReposition(node.GlobalPosition, HoverTipAlignment.Left);
				_textHoverTipContainer.GlobalPosition = node.GlobalPosition + new Vector2(node.Size.X, 0f) * node.Scale;
				break;
			case HoverTipAlignment.Center:
				base.GlobalPosition = node.GlobalPosition + Vector2.Down * node.Size.Y * 1.5f;
				_cardHoverTipContainer.GlobalPosition += Vector2.Down * _textHoverTipContainer.Size.Y;
				_cardHoverTipContainer.LayoutResizeAndReposition(_cardHoverTipContainer.GlobalPosition, alignment);
				break;
			}
		}
		CorrectVerticalOverflow();
		CorrectHorizontalOverflow();
	}

	public void SetAlignmentForRelic(NRelic relic)
	{
		HoverTipAlignment hoverTipAlignment = HoverTip.GetHoverTipAlignment(relic);
		Vector2 vector = relic.Icon.Size * relic.GetGlobalTransform().Scale;
		_textHoverTipContainer.GlobalPosition = relic.GlobalPosition + Vector2.Down * (vector.Y + 10f);
		if (hoverTipAlignment == HoverTipAlignment.Left)
		{
			_textHoverTipContainer.Position += Vector2.Left * (_textHoverTipContainer.Size.X - vector.X);
		}
		_cardHoverTipContainer.LayoutResizeAndReposition(_textHoverTipContainer.GlobalPosition + Vector2.Down * _textHoverTipContainer.Size.Y, hoverTipAlignment);
		if (hoverTipAlignment == HoverTipAlignment.Left)
		{
			_cardHoverTipContainer.GlobalPosition = new Vector2(_textHoverTipContainer.GlobalPosition.X, _cardHoverTipContainer.GlobalPosition.Y);
		}
		float y = NGame.Instance.GetViewportRect().Size.Y;
		if (relic.GlobalPosition.Y > y * 0.75f)
		{
			_textHoverTipContainer.GlobalPosition = relic.GlobalPosition + Vector2.Up * _textHoverTipContainer.Size.Y;
		}
		CorrectVerticalOverflow();
		CorrectHorizontalOverflow();
		if (_textHoverTipContainer.GetRect().Intersects(_cardHoverTipContainer.GetRect()))
		{
			if (hoverTipAlignment == HoverTipAlignment.Left)
			{
				_cardHoverTipContainer.GlobalPosition = _textHoverTipContainer.GlobalPosition + _textHoverTipContainer.Size.X * Vector2.Right;
			}
			else
			{
				_cardHoverTipContainer.GlobalPosition = _textHoverTipContainer.GlobalPosition + _cardHoverTipContainer.Size.X * Vector2.Left;
			}
			CorrectVerticalOverflow();
		}
	}

	public void SetAlignmentForCardHolder(NCardHolder holder)
	{
		HoverTipAlignment hoverTipAlignment = HoverTip.GetHoverTipAlignment(holder);
		_textHoverTipContainer.Position = Vector2.Zero;
		Control hitbox = holder.Hitbox;
		if (hoverTipAlignment == HoverTipAlignment.Left)
		{
			_textHoverTipContainer.GlobalPosition = hitbox.GlobalPosition;
			_textHoverTipContainer.Position += Vector2.Left * _textHoverTipContainer.Size.X - new Vector2(10f, 0f);
			_textHoverTipContainer.ReverseFill = true;
			_cardHoverTipContainer.LayoutResizeAndReposition(hitbox.GlobalPosition + new Vector2(hitbox.Size.X, 0f) * hitbox.Scale, HoverTipAlignment.Right);
		}
		else
		{
			Vector2 globalPosition = hitbox.GlobalPosition;
			if (holder.CardModel != null && (holder.CardModel.CurrentStarCost > 0 || holder.CardModel.HasStarCostX))
			{
				globalPosition += Vector2.Left * 15f;
			}
			_cardHoverTipContainer.LayoutResizeAndReposition(globalPosition, HoverTipAlignment.Left);
			_textHoverTipContainer.GlobalPosition = hitbox.GlobalPosition + new Vector2(hitbox.Size.X + 10f, 0f) * hitbox.Scale * holder.Scale;
		}
		CorrectVerticalOverflow();
		CorrectHorizontalOverflow();
		SetFollowOwner();
	}

	private void CorrectVerticalOverflow()
	{
		float y = NGame.Instance.GetViewportRect().Size.Y;
		if (_textHoverTipContainer.GlobalPosition.Y + _textHoverTipContainer.Size.Y > y)
		{
			_textHoverTipContainer.GlobalPosition = new Vector2(_textHoverTipContainer.GlobalPosition.X, y - _textHoverTipContainer.Size.Y);
		}
		if (_cardHoverTipContainer.GlobalPosition.Y + _cardHoverTipContainer.Size.Y > y)
		{
			_cardHoverTipContainer.GlobalPosition = new Vector2(_cardHoverTipContainer.GlobalPosition.X, y - _cardHoverTipContainer.Size.Y);
		}
	}

	private void CorrectHorizontalOverflow()
	{
		float x = NGame.Instance.GetViewportRect().Size.X;
		Vector2 globalPosition = _cardHoverTipContainer.GlobalPosition;
		float x2 = _cardHoverTipContainer.Size.X;
		Vector2 globalPosition2 = _textHoverTipContainer.GlobalPosition;
		float x3 = _textHoverTipContainer.Size.X;
		if (globalPosition.X + x2 <= x && globalPosition2.X + x3 > x)
		{
			float x4 = globalPosition.X - x3;
			_textHoverTipContainer.GlobalPosition = new Vector2(x4, globalPosition.Y);
		}
		else if (globalPosition.X + x2 > x || globalPosition2.X + x3 > x)
		{
			float x5 = globalPosition2.X + x3 - x2;
			_cardHoverTipContainer.GlobalPosition = new Vector2(x5, globalPosition.Y);
			_textHoverTipContainer.GlobalPosition += Vector2.Left * x2;
		}
		else if (globalPosition.X < 0f || globalPosition2.X < 0f)
		{
			float x6 = globalPosition2.X;
			_cardHoverTipContainer.GlobalPosition = new Vector2(x6, globalPosition.Y);
			_textHoverTipContainer.GlobalPosition += Vector2.Right * x2;
		}
	}

	public static void Clear()
	{
		foreach (Control key in _activeHoverTips.Keys)
		{
			Remove(key);
		}
	}

	public static void Remove(Control owner)
	{
		if (_activeHoverTips.TryGetValue(owner, out NHoverTipSet value))
		{
			value.QueueFreeSafely();
			_activeHoverTips.Remove(owner);
		}
	}

	public void SetExtraFollowOffset(Vector2 offset)
	{
		_extraOffset = offset;
	}
}
