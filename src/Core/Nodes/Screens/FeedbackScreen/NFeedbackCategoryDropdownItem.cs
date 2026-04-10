using Godot;
using MegaCrit.Sts2.Core.Nodes.CommonUi;

namespace MegaCrit.Sts2.Core.Nodes.Screens.FeedbackScreen;

public partial class NFeedbackCategoryDropdownItem : NDropdownItem
{
	public int CategoryIndex { get; private set; }

	public void Init(int categoryIndex, string localizedCategory)
	{
		CategoryIndex = categoryIndex;
		base.Text = localizedCategory;
	}
}
