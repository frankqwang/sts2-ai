using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Combat.History;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.Nodes.Debug;

public partial class NCommandHistory : Panel
{
	private static NCommandHistory? _instance;

	private RichTextLabel _outputBuffer;

	public override void _Ready()
	{
		if (_instance != null)
		{
			this.QueueFreeSafely();
			return;
		}
		_instance = this;
		HideConsole();
		_outputBuffer = GetNode<RichTextLabel>("OutputContainer/OutputBuffer");
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (!(inputEvent is InputEventKey { Pressed: not false } inputEventKey))
		{
			return;
		}
		if (inputEventKey.Keycode == Key.Plus)
		{
			if (base.Visible)
			{
				HideConsole();
			}
			else
			{
				ShowConsole();
			}
		}
		if (base.Visible && inputEventKey.Keycode == Key.Escape)
		{
			HideConsole();
		}
	}

	public void SetBackgroundColor(Color color)
	{
		base.Modulate = color;
	}

	public void ShowConsole()
	{
		base.Visible = true;
		CombatManager.Instance.History.Changed += Refresh;
		Refresh();
	}

	public void HideConsole()
	{
		CombatManager.Instance.History.Changed -= Refresh;
		base.Visible = false;
	}

	public static string GetHistory()
	{
		if (_instance != null)
		{
			return GetText();
		}
		return string.Empty;
	}

	private void Refresh()
	{
		_outputBuffer.Text = GetText();
	}

	private static string GetText()
	{
		return string.Join('\n', CombatManager.Instance.History.Entries.Select((CombatHistoryEntry e) => e.HumanReadableString));
	}
}
