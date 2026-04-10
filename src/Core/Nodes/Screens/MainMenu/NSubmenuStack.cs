using System;
using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;

namespace MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;

public abstract partial class NSubmenuStack : Control
{
	[Signal]
	public delegate void StackModifiedEventHandler();

	private readonly Stack<NSubmenu> _submenus = new Stack<NSubmenu>();

	private NMainMenu? _mainMenu;

	public bool SubmenusOpen => _submenus.Count > 0;

	public void InitializeForMainMenu(NMainMenu mainMenu)
	{
		_mainMenu = mainMenu;
	}

	public abstract T PushSubmenuType<T>() where T : NSubmenu;

	public abstract T GetSubmenuType<T>() where T : NSubmenu;

	public abstract NSubmenu PushSubmenuType(Type type);

	public abstract NSubmenu GetSubmenuType(Type type);

	public void Push(NSubmenu screen)
	{
		if (_submenus.Count > 0)
		{
			NSubmenu nSubmenu = _submenus.Peek();
			nSubmenu.Visible = false;
			nSubmenu.MouseFilter = MouseFilterEnum.Ignore;
		}
		screen.SetStack(this);
		_submenus.Push(screen);
		screen.OnSubmenuOpened();
		screen.Visible = true;
		screen.MouseFilter = MouseFilterEnum.Stop;
		_mainMenu?.EnableBackstop();
		ActiveScreenContext.Instance.Update();
		EmitSignal(SignalName.StackModified);
	}

	public void Pop()
	{
		NSubmenu nSubmenu = _submenus.Pop();
		nSubmenu.Visible = false;
		nSubmenu.MouseFilter = MouseFilterEnum.Ignore;
		nSubmenu.OnSubmenuClosed();
		if (_submenus.Count > 0)
		{
			NSubmenu nSubmenu2 = _submenus.Peek();
			nSubmenu2.Visible = true;
			nSubmenu2.MouseFilter = MouseFilterEnum.Stop;
		}
		else
		{
			HideBackstop();
		}
		ActiveScreenContext.Instance.Update();
		EmitSignal(SignalName.StackModified);
	}

	private void ShowBackstop()
	{
		_mainMenu?.EnableBackstop();
	}

	private void HideBackstop()
	{
		_mainMenu?.DisableBackstop();
	}

	public NSubmenu? Peek()
	{
		if (!_submenus.TryPeek(out NSubmenu result))
		{
			return null;
		}
		return result;
	}
}
