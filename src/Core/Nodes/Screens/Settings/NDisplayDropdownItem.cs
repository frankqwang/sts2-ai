using Godot;
using MegaCrit.Sts2.Core.Nodes.CommonUi;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NDisplayDropdownItem : NDropdownItem
{
	public int displayIndex;

	public void Init(int setIndex)
	{
		displayIndex = setIndex;
		string textAutoSize = $"Monitor ({displayIndex})";
		_label.SetTextAutoSize(textAutoSize);
	}
}
