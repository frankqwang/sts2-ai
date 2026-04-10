using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.CommonUi;

public partial class NDropdownItem : NButton
{
	[Signal]
	public delegate void SelectedEventHandler(NDropdownItem cardHolder);

	private ColorRect _highlight;

	protected MegaLabel _label;

	protected MegaRichTextLabel? _richLabel;

	public string Text
	{
		get
		{
			return _label.Text;
		}
		set
		{
			_label.SetTextAutoSize(value);
		}
	}

	public override void _Ready()
	{
		ConnectSignals();
		_highlight = GetNode<ColorRect>("Highlight");
		_label = GetNodeOrNull<MegaLabel>("Label");
		_richLabel = GetNodeOrNull<MegaRichTextLabel>("RichLabel");
	}

	protected override void OnFocus()
	{
		_highlight.Visible = true;
	}

	protected override void OnUnfocus()
	{
		_highlight.Visible = false;
	}

	protected override void OnPress()
	{
		_highlight.Visible = false;
	}

	protected sealed override void OnRelease()
	{
		_highlight.Visible = true;
		EmitSignal(SignalName.Selected, this);
	}

	public void UnhoverSelection()
	{
		OnUnfocus();
	}
}
