using Godot;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public abstract partial class NSettingsDropdown : NDropdown
{
	private NSelectionReticle _selectionReticle;

	protected override void ConnectSignals()
	{
		base.ConnectSignals();
		_selectionReticle = GetNode<NSelectionReticle>("SelectionReticle");
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_currentOptionHighlight.Modulate = new Color("3C5B6B");
		if (NControllerManager.Instance.IsUsingController)
		{
			_selectionReticle.OnSelect();
		}
	}

	protected override void OnUnfocus()
	{
		_selectionReticle.OnDeselect();
		_currentOptionHighlight.Modulate = new Color("2C434F");
	}
}
