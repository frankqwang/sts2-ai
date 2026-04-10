using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NSelectedHandCardContainer : Control
{
	public NPlayerHand? Hand { get; set; }

	public List<NSelectedHandCardHolder> Holders => GetChildren().OfType<NSelectedHandCardHolder>().ToList();

	public override void _Ready()
	{
		Connect(Control.SignalName.FocusEntered, Callable.From(OnFocus));
	}

	public NSelectedHandCardHolder Add(NHandCardHolder originalHolder)
	{
		NCard cardNode = originalHolder.CardNode;
		Vector2 globalPosition = cardNode.GlobalPosition;
		NSelectedHandCardHolder nSelectedHandCardHolder = NSelectedHandCardHolder.Create(originalHolder);
		nSelectedHandCardHolder.Connect(NCardHolder.SignalName.Pressed, Callable.From<NCardHolder>(DeselectHolder), 4u);
		this.AddChildSafely(nSelectedHandCardHolder);
		RefreshHolderPositions();
		cardNode.GlobalPosition = globalPosition;
		return nSelectedHandCardHolder;
	}

	private void RefreshHolderPositions()
	{
		int count = Holders.Count;
		base.FocusMode = (FocusModeEnum)((count > 0) ? 2 : 0);
		if (count != 0)
		{
			float x = Holders.First().Size.X;
			float num = (0f - x) * (float)(count - 1) / 2f;
			for (int i = 0; i < count; i++)
			{
				Holders[i].Position = new Vector2(num, 0f);
				num += x;
				Holders[i].FocusNeighborLeft = ((i > 0) ? Holders[i - 1].GetPath() : Holders[Holders.Count - 1].GetPath());
				Holders[i].FocusNeighborRight = ((i < Holders.Count - 1) ? Holders[i + 1].GetPath() : Holders[0].GetPath());
			}
		}
	}

	private void DeselectHolder(NCardHolder holder)
	{
		NSelectedHandCardHolder nSelectedHandCardHolder = (NSelectedHandCardHolder)holder;
		NCard cardNode = nSelectedHandCardHolder.CardNode;
		Hand.DeselectCard(cardNode);
		this.RemoveChildSafely(nSelectedHandCardHolder);
		nSelectedHandCardHolder.QueueFreeSafely();
		RefreshHolderPositions();
	}

	public void DeselectCard(CardModel card)
	{
		NSelectedHandCardHolder holder = Holders.First((NSelectedHandCardHolder child) => child.CardNode.Model == card);
		DeselectHolder(holder);
	}

	private void OnFocus()
	{
		Holders.FirstOrDefault()?.TryGrabFocus();
	}
}
