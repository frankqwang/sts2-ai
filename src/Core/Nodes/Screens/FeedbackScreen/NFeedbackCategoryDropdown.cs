using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Extensions;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.FeedbackScreen;

public partial class NFeedbackCategoryDropdown : NDropdown
{
	[Export(PropertyHint.None, "")]
	private PackedScene _dropdownItemScene;

	private NSelectionReticle _selectionReticle;

	private static readonly string[] _categories = new string[3] { "bug", "balance", "feedback" };

	private static readonly LocString[] _categoryLoc = new LocString[3]
	{
		new LocString("settings_ui", "FEEDBACK_CATEGORY.bug"),
		new LocString("settings_ui", "FEEDBACK_CATEGORY.balance"),
		new LocString("settings_ui", "FEEDBACK_CATEGORY.feedback")
	};

	private int _currentCategoryIndex = _categories.IndexOf("feedback");

	public string CurrentCategory => _categories[_currentCategoryIndex];

	public override void _Ready()
	{
		ConnectSignals();
		_currentOptionHighlight = GetNode<Panel>("%Highlight");
		_currentOptionLabel = GetNode<MegaLabel>("%Label");
		PopulateOptions();
		_currentOptionLabel.SetTextAutoSize(_categoryLoc[_currentCategoryIndex].GetFormattedText());
		_selectionReticle = GetNode<NSelectionReticle>("SelectionReticle");
	}

	protected override void OnFocus()
	{
		_currentOptionHighlight.Modulate = new Color("afcdde");
		if (NControllerManager.Instance.IsUsingController)
		{
			_selectionReticle.OnSelect();
		}
	}

	protected override void OnUnfocus()
	{
		_selectionReticle.OnDeselect();
		_currentOptionHighlight.Modulate = Colors.White;
	}

	private void PopulateOptions()
	{
		Control node = GetNode<Control>("DropdownContainer/VBoxContainer");
		foreach (Node child in node.GetChildren())
		{
			node.RemoveChildSafely(child);
			child.QueueFreeSafely();
		}
		for (int i = 0; i < _categories.Length; i++)
		{
			NFeedbackCategoryDropdownItem nFeedbackCategoryDropdownItem = _dropdownItemScene.Instantiate<NFeedbackCategoryDropdownItem>(PackedScene.GenEditState.Disabled);
			node.AddChildSafely(nFeedbackCategoryDropdownItem);
			nFeedbackCategoryDropdownItem.Connect(NDropdownItem.SignalName.Selected, Callable.From<NDropdownItem>(OnDropdownItemSelected));
			nFeedbackCategoryDropdownItem.Init(i, _categoryLoc[i].GetFormattedText());
		}
		node.GetParent<NDropdownContainer>().RefreshLayout();
	}

	private void OnDropdownItemSelected(NDropdownItem item)
	{
		NFeedbackCategoryDropdownItem nFeedbackCategoryDropdownItem = (NFeedbackCategoryDropdownItem)item;
		if (nFeedbackCategoryDropdownItem.CategoryIndex != _currentCategoryIndex)
		{
			CloseDropdown();
			_currentCategoryIndex = nFeedbackCategoryDropdownItem.CategoryIndex;
			_currentOptionLabel.SetTextAutoSize(_categoryLoc[_currentCategoryIndex].GetFormattedText());
		}
	}
}
