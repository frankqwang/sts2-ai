using System;
using System.Linq;
using Godot;

namespace MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;

[Tool]
public partial class NJoinFriendScreenButtonLayout : Container
{
	public override void _Notification(int what)
	{
		if ((long)what == 51)
		{
			LayoutChildren();
		}
	}

	private void LayoutChildren()
	{
		Control[] array = GetChildren().OfType<Control>().ToArray();
		if (array.Length != 0)
		{
			Vector2 size = array[0].Size;
			int num = (int)(base.Size.Y / size.Y);
			int num2 = (int)Math.Ceiling((float)array.Length / (float)num);
			float num3 = (base.Size.X - (float)num2 * size.X) * 0.5f;
			for (int i = 0; i < array.Length; i++)
			{
				int num4 = i / num2;
				int num5 = i - num4 * num2;
				array[i].Position = new Vector2(num5, num4) * size + Vector2.Right * num3;
			}
		}
	}
}
