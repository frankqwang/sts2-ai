using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Timeline;

public partial class NEpochReminderText : Control
{
	private MegaRichTextLabel _label;

	private LocString _loc;

	private Tween? _tween;

	private Control _vfxHolder;

	public override void _Ready()
	{
		_label = GetNode<MegaRichTextLabel>("%ReminderLabel");
	}

	public void AnimateIn()
	{
		int discoveredEpochCount = SaveManager.Instance.GetDiscoveredEpochCount();
		if (discoveredEpochCount != 0)
		{
			base.Visible = true;
			_loc = new LocString("timeline", "REMINDER_TEXT");
			_loc.AddObj("RevealableEpochCount", discoveredEpochCount);
			_label.Text = _loc.GetFormattedText();
			_tween?.Kill();
			_tween = CreateTween().SetLoops();
			_tween.TweenProperty(_label, "modulate:a", 1f, 0.8);
			_tween.TweenProperty(_label, "modulate:a", 0.25f, 0.8);
		}
	}

	public void AnimateOut()
	{
		_tween?.Kill();
		_tween = CreateTween();
		_tween.TweenProperty(_label, "modulate:a", 0f, 0.25);
	}
}
