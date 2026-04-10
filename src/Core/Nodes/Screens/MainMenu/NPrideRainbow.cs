using System;
using Godot;

namespace MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;

public partial class NPrideRainbow : Control
{
	public override void _Ready()
	{
		base.Visible = DateTime.Now.Month == 6;
	}
}
