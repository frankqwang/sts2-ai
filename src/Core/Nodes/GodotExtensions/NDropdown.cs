using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Debug;

namespace MegaCrit.Sts2.Core.Nodes.GodotExtensions;

public partial class NDropdown : NClickableControl
{
	private Control _dropdownContainer;

	protected Control _dropdownItems;

	private NButton _dismisser;

	protected MegaLabel _currentOptionLabel;

	protected Control _currentOptionHighlight;

	private bool _isHovered;

	private bool _isOpen;

	public override void _Ready()
	{
		if (GetType() != typeof(NDropdown))
		{
			throw new InvalidOperationException("Don't call base._Ready(). Use ConnectSignals() instead");
		}
		ConnectSignals();
	}

	protected override void ConnectSignals()
	{
		base.ConnectSignals();
		_currentOptionHighlight = GetNode<Control>("%Highlight");
		_currentOptionLabel = GetNode<MegaLabel>("%Label");
		_dropdownContainer = GetNode<Control>("%DropdownContainer");
		_dropdownItems = _dropdownContainer.GetNode<Control>("VBoxContainer");
		_dismisser = GetNode<NButton>("%Dismisser");
		_dismisser.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnDismisserClicked));
		Connect(CanvasItem.SignalName.VisibilityChanged, Callable.From(OnVisibilityChange));
	}

	private void OnVisibilityChange()
	{
		if (!IsVisibleInTree() && _isOpen)
		{
			CloseDropdown();
		}
	}

	protected void ClearDropdownItems()
	{
		foreach (Node child in _dropdownItems.GetChildren())
		{
			_dropdownItems.RemoveChildSafely(child);
			child.QueueFreeSafely();
		}
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (!IsVisibleInTree() || !_isEnabled || NDevConsole.Instance.Visible)
		{
			return;
		}
		Viewport viewport = GetViewport();
		if (viewport != null)
		{
			Control control = viewport.GuiGetFocusOwner();
			bool flag = ((control is TextEdit || control is LineEdit) ? true : false);
			if (!flag && inputEvent.IsActionPressed(MegaInput.cancel) && _isOpen)
			{
				CloseDropdown();
				viewport.SetInputAsHandled();
			}
		}
	}

	private void OnDismisserClicked(NButton obj)
	{
		CloseDropdown();
	}

	protected override void OnRelease()
	{
		base.OnRelease();
		if (_isOpen)
		{
			Log.Info("Closing dropdown because you clicked on the main dropdown button.");
			CloseDropdown();
		}
		else
		{
			OpenDropdown();
		}
	}

	private void OpenDropdown()
	{
		_dropdownContainer.Visible = true;
		_dismisser.Visible = true;
		_isOpen = true;
		GetParent().MoveChild(this, GetParent().GetChildCount());
		List<NDropdownItem> list = _dropdownItems.GetChildren().OfType<NDropdownItem>().ToList();
		for (int i = 0; i < list.Count; i++)
		{
			list[i].UnhoverSelection();
			list[i].FocusNeighborLeft = list[i].GetPath();
			list[i].FocusNeighborRight = list[i].GetPath();
			list[i].FocusNeighborTop = ((i > 0) ? list[i - 1].GetPath() : list[i].GetPath());
			list[i].FocusNeighborBottom = ((i < list.Count - 1) ? list[i + 1].GetPath() : list[i].GetPath());
			list[i].FocusMode = FocusModeEnum.All;
		}
		list.FirstOrDefault()?.TryGrabFocus();
	}

	protected void CloseDropdown()
	{
		_dismisser.Visible = false;
		_dropdownContainer.Visible = false;
		_isOpen = false;
		this.TryGrabFocus();
	}
}
