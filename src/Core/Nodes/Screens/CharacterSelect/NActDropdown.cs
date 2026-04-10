using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using Godot;
using MegaCrit.Sts2.Core.Extensions;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;

public partial class NActDropdown : NDropdown
{
	private static readonly string[] _options = new string[3] { "random", "overgrowth", "underdocks" };

	private int _currentOptionIndex = _options.IndexOf("random");

	public string CurrentOption => _options[_currentOptionIndex];

	public override void _Ready()
	{
		ConnectSignals();
		PopulateOptions();
	}

	protected override void OnFocus()
	{
		_currentOptionHighlight.Modulate = new Color("afcdde");
	}

	protected override void OnUnfocus()
	{
		_currentOptionHighlight.Modulate = Colors.White;
	}

	private void PopulateOptions()
	{
		List<NDropdownItem> list = GetDropdownItems().ToList();
		for (int i = 0; i < _options.Length; i++)
		{
			NDropdownItem nDropdownItem = list[i];
			string text = _options[i];
			nDropdownItem.Connect(NDropdownItem.SignalName.Selected, Callable.From<NDropdownItem>(OnDropdownItemSelected));
			DefaultInterpolatedStringHandler defaultInterpolatedStringHandler = new DefaultInterpolatedStringHandler(0, 2);
			defaultInterpolatedStringHandler.AppendFormatted(char.ToUpperInvariant(text[0]));
			string text2 = text;
			defaultInterpolatedStringHandler.AppendFormatted(text2.Substring(1, text2.Length - 1));
			nDropdownItem.Text = defaultInterpolatedStringHandler.ToStringAndClear();
		}
		GetDropdownContainer().GetParent<NDropdownContainer>().RefreshLayout();
	}

	private void OnDropdownItemSelected(NDropdownItem item)
	{
		CloseDropdown();
		_currentOptionIndex = GetDropdownItems().ToList().IndexOf(item);
		_currentOptionLabel.SetTextAutoSize(item.Text);
	}

	private Control GetDropdownContainer()
	{
		return GetNode<Control>("DropdownContainer/VBoxContainer");
	}

	private IEnumerable<NDropdownItem> GetDropdownItems()
	{
		return GetDropdownContainer().GetChildren().OfType<NDropdownItem>();
	}
}
