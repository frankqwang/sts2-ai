using Godot;

namespace MegaCrit.Sts2.addons.mega_text;

public partial class MegaRichTextLabel : RichTextLabel
{
    public object? AutowrapMode { get; set; }

    public void SetTextAutoSize(string text)
    {
        Text = text;
    }
}

public partial class MegaLabel : Label
{
    public void SetTextAutoSize(string text)
    {
        Text = text;
    }
}
