using System.Collections.Generic;
using System.Data;
using Godot;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Pooling;
using MegaCrit.Sts2.Core.Nodes.Screens.CardLibrary;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Cards.Holders;

public partial class NGridCardHolder : NCardHolder, IPoolable
{
	private CardModel _baseCard;

	private CardModel? _upgradedCard;

	private bool _isPreviewingUpgrade;

	private static string ScenePath => SceneHelper.GetScenePath("cards/holders/grid_card_holder");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	public NCardLibraryStats? CardLibraryStats { get; private set; }

	public override CardModel CardModel => _baseCard;

	public override bool IsShowingUpgradedCard
	{
		get
		{
			if (!_isPreviewingUpgrade)
			{
				return base.IsShowingUpgradedCard;
			}
			return true;
		}
	}

	public static void InitPool()
	{
		NodePool.Init<NGridCardHolder>(ScenePath, 30);
	}

	public static NGridCardHolder? Create(NCard cardNode)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NGridCardHolder nGridCardHolder = NodePool.Get<NGridCardHolder>();
		nGridCardHolder.SetCard(cardNode);
		nGridCardHolder.UpdateCardModel();
		nGridCardHolder.UpdateName();
		nGridCardHolder.Scale = nGridCardHolder.SmallScale;
		return nGridCardHolder;
	}

	private void UpdateCardModel()
	{
		CardModel cardModel = (_baseCard = base.CardNode.Model);
		if (cardModel.IsUpgradable)
		{
			_upgradedCard = (CardModel)cardModel.MutableClone();
			_upgradedCard.UpgradeInternal();
			if (IsNodeReady())
			{
				bool isPreviewingUpgrade = _isPreviewingUpgrade;
				_isPreviewingUpgrade = false;
				SetIsPreviewingUpgrade(isPreviewingUpgrade);
			}
		}
	}

	public void OnInstantiated()
	{
	}

	public override void _Ready()
	{
		bool isPreviewingUpgrade = _isPreviewingUpgrade;
		_isPreviewingUpgrade = false;
		SetIsPreviewingUpgrade(isPreviewingUpgrade);
		ConnectSignals();
	}

	public void EnsureCardLibraryStatsExists()
	{
		if (CardLibraryStats == null)
		{
			CardLibraryStats = NCardLibraryStats.Create();
			this.AddChildSafely(CardLibraryStats);
		}
	}

	protected override void OnCardReassigned()
	{
		UpdateCardModel();
		UpdateName();
	}

	protected override void SetCard(NCard node)
	{
		base.SetCard(node);
		if (CardLibraryStats != null)
		{
			MoveChild(CardLibraryStats, GetChildCount() - 1);
		}
	}

	private void UpdateName()
	{
		base.Name = $"GridCardHolder-{base.CardNode.Model.Id}";
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		MoveToFront();
	}

	public void SetIsPreviewingUpgrade(bool showUpgradePreview)
	{
		if (!base.Visible)
		{
			return;
		}
		if (!_baseCard.IsUpgradable && showUpgradePreview)
		{
			throw new InvalidExpressionException($"{_baseCard.Id} is not upgradable.");
		}
		if (_isPreviewingUpgrade != showUpgradePreview)
		{
			if (showUpgradePreview && _upgradedCard != null)
			{
				base.CardNode.Model = _upgradedCard;
				base.CardNode.ShowUpgradePreview();
			}
			else
			{
				base.CardNode.Model = _baseCard;
				base.CardNode.UpdateVisuals(base.CardNode.DisplayingPile, CardPreviewMode.Normal);
			}
			_isPreviewingUpgrade = showUpgradePreview;
		}
	}

	public override void _ExitTree()
	{
		if (IsAncestorOf(base.CardNode))
		{
			base.CardNode?.QueueFreeSafely();
		}
		base.CardNode = null;
	}

	public void OnReturnedFromPool()
	{
		if (IsNodeReady())
		{
			base.Position = Vector2.Zero;
			base.Rotation = 0f;
			base.Scale = Vector2.One;
			base.Modulate = Colors.White;
			base.Visible = true;
			SetClickable(isClickable: true);
			base.Hitbox.MouseDefaultCursorShape = CursorShape.Arrow;
			_isPreviewingUpgrade = false;
			if (CardLibraryStats != null)
			{
				CardLibraryStats.Visible = false;
				CardLibraryStats.Modulate = Colors.White;
			}
		}
	}

	public void OnFreedToPool()
	{
	}
}
