using System;
using Godot;
using MegaCrit.Sts2.Core.Nodes.Cards;

namespace MegaCrit.Sts2.Core.Nodes.CommonUi;

public partial class NGridCardPreviewContainer : Control
{
	private int? _forcedMaxColumns;

	public override void _Ready()
	{
		Connect(Node.SignalName.ChildEnteredTree, Callable.From<Node>(ReformatElements));
		Connect(Node.SignalName.ChildExitingTree, Callable.From<Node>(CheckAnyChildrenPresent));
	}

	public void ForceMaxColumnsUntilEmpty(int maxColumns)
	{
		_forcedMaxColumns = maxColumns;
	}

	private void ReformatElements(Node _)
	{
		Vector2 vector = base.Size / 2f + Vector2.Down * 50f;
		int childCount = GetChildCount();
		int num = Mathf.FloorToInt(GetViewportRect().Size.X / (NCard.defaultSize.X + 25f));
		int num2 = Mathf.CeilToInt((float)childCount / (float)num);
		if (_forcedMaxColumns.HasValue)
		{
			num = Math.Min(num, _forcedMaxColumns.Value);
		}
		for (int i = 0; i < childCount; i++)
		{
			int num3 = i / num;
			int num4 = i % num;
			int num5 = Math.Min(num, childCount - num3 * num);
			float num6 = (float)(-(num2 - 1)) * (NCard.defaultSize.Y + 25f) * 0.5f;
			float num7 = (float)(-(num5 - 1)) * (NCard.defaultSize.X + 25f) * 0.5f;
			Vector2 vector2 = vector;
			Vector2 position = vector2 + new Vector2(num7 + (NCard.defaultSize.X + 25f) * (float)num4, num6 + (NCard.defaultSize.Y + 25f) * (float)num3);
			Node child = GetChild(i);
			if (child is Node2D node2D)
			{
				node2D.Position = position;
			}
			else
			{
				((Control)child).Position = position;
			}
		}
	}

	private void CheckAnyChildrenPresent(Node _)
	{
		if (GetChildCount() == 0)
		{
			_forcedMaxColumns = null;
		}
	}
}
