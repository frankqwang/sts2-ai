using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Localization.Fonts;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Platform;

namespace MegaCrit.Sts2.Core.Nodes.GodotExtensions;

public partial class NMegaTextEdit : TextEdit
{
	private NSelectionReticle? _selectionReticle;

	private bool _isEditing;

	public bool IsEditing()
	{
		return _isEditing;
	}

	public override void _Ready()
	{
		RefreshFont();
		Connect(Control.SignalName.FocusExited, Callable.From(StopEditing));
		if (HasNode("SelectionReticle"))
		{
			_selectionReticle = GetNode<NSelectionReticle>("SelectionReticle");
		}
		Connect(Control.SignalName.FocusEntered, Callable.From(OnFocus));
		Connect(Control.SignalName.FocusExited, Callable.From(OnUnfocus));
		Connect(Control.SignalName.MouseEntered, Callable.From(OnFocus));
		Connect(Control.SignalName.MouseExited, Callable.From(OnUnfocus));
	}

	public void RefreshFont()
	{
		this.ApplyLocaleFontSubstitution(FontType.Regular, ThemeConstants.TextEdit.font);
	}

	private void OnFocus()
	{
		if (NControllerManager.Instance.IsUsingController)
		{
			_selectionReticle?.OnSelect();
		}
	}

	private void OnUnfocus()
	{
		_selectionReticle?.OnDeselect();
	}

	public override void _GuiInput(InputEvent inputEvent)
	{
		base._GuiInput(inputEvent);
		if (inputEvent is InputEventMouseButton inputEventMouseButton && inputEventMouseButton.ButtonIndex == MouseButton.Left && inputEventMouseButton.IsPressed())
		{
			OpenKeyboard();
		}
		if (inputEvent.IsActionPressed(MegaInput.select))
		{
			OpenKeyboard();
		}
		if (inputEvent.IsActionPressed(MegaInput.cancel) && IsEditing())
		{
			StopEditing();
			GetViewport()?.SetInputAsHandled();
		}
	}

	private void OpenKeyboard()
	{
		this.TryGrabFocus();
		_isEditing = true;
		PlatformUtil.OpenVirtualKeyboard();
	}

	private void StopEditing()
	{
		_isEditing = false;
		PlatformUtil.CloseVirtualKeyboard();
	}
}
