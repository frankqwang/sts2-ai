using Godot;

namespace MegaCrit.Sts2.Core.Nodes.CommonUi;

public partial class NCardPreviewContainer : Control
{
	public override void _Ready()
	{
		Connect(Node.SignalName.ChildEnteredTree, Callable.From<Node>(ReformatElements));
	}

	private void ReformatElements(Node _)
	{
		Vector2 vector = base.Size / 2f + Vector2.Down * 50f;
		int childCount = GetChildCount();
		for (int i = 0; i < childCount; i++)
		{
			float num = (float)(-(childCount - 1) * 325) * 0.5f;
			Vector2 vector2 = vector;
			Vector2 position = vector2 + Vector2.Right * (num + (float)(325 * i));
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
}
