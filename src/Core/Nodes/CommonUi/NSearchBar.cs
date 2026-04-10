using System;
using System.CodeDom.Compiler;
using System.Text.RegularExpressions;
using System.Text.RegularExpressions.Generated;
using Godot;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.CommonUi;

public partial class NSearchBar : Control
{
	[Signal]
	public delegate void QueryChangedEventHandler(string query);

	[Signal]
	public delegate void QuerySubmittedEventHandler(string query);

	private LineEdit _textArea;

	private NButton _clearButton;

	public string Text => _textArea.Text;

	public LineEdit TextArea => _textArea;

	[GeneratedRegex("[\\t\\r\\n]")]
	private static partial Regex NonSpaceWhitespaceCharacters();

	[GeneratedRegex("\\s{2,}")]
	private static partial Regex ConsecutiveSpaces();

	[GeneratedRegex("<.*?>")]
	private static partial Regex HtmlTags();

	public override void _Ready()
	{
		_textArea = GetNode<LineEdit>("TextArea");
		_textArea.Connect(LineEdit.SignalName.TextChanged, Callable.From<string>(TextUpdated));
		_textArea.Connect(LineEdit.SignalName.TextSubmitted, Callable.From<string>(TextSubmitted));
		_textArea.SetPlaceholder(new LocString("card_library", "SEARCH_PLACEHOLDER").GetRawText());
		_clearButton = GetNode<NButton>("ClearButton");
		_clearButton.Connect(NClickableControl.SignalName.Released, Callable.From((Action<NButton>)ClearText));
	}

	private void TextUpdated(string _)
	{
		EmitSignal(SignalName.QueryChanged, _textArea.Text);
	}

	private void TextSubmitted(string _)
	{
		EmitSignal(SignalName.QuerySubmitted, _textArea.Text);
	}

	private void ClearText(NButton _)
	{
		ClearText();
	}

	public void ClearText()
	{
		_textArea.TryGrabFocus();
		if (!string.IsNullOrWhiteSpace(_textArea.Text))
		{
			_textArea.Text = "";
			EmitSignal(SignalName.QueryChanged, _textArea.Text);
		}
	}

	public static string Normalize(string text)
	{
		string input = NonSpaceWhitespaceCharacters().Replace(text.Trim(), " ");
		return ConsecutiveSpaces().Replace(input, " ").ToLowerInvariant();
	}

	public static string RemoveHtmlTags(string text)
	{
		return HtmlTags().Replace(text, string.Empty);
	}
}
