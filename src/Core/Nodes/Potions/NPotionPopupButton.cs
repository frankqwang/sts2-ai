using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;

namespace MegaCrit.Sts2.Core.Nodes.Potions;

public partial class NPotionPopupButton : NButton
{
	private TextureRect _background;

	private MegaLabel _label;

	private Tween? _currentTween;

	public override void _Ready()
	{
		ConnectSignals();
		_background = GetNode<TextureRect>("Background");
		_label = GetNode<MegaLabel>("Label");
		_background.Modulate = Colors.Transparent;
	}

	public void SetLocKey(string locEntryKey)
	{
		_label.SetTextAutoSize(new LocString("gameplay_ui", locEntryKey).GetFormattedText());
	}

	protected override void OnFocus()
	{
		_currentTween?.Kill();
		_currentTween = CreateTween();
		_currentTween.TweenProperty(_background, "modulate:a", 0.25f, 0.15000000596046448);
	}

	protected override void OnUnfocus()
	{
		NHoverTipSet.Remove(this);
		_currentTween?.Kill();
		_currentTween = CreateTween();
		_currentTween.TweenProperty(_background, "modulate:a", 0f, 0.15000000596046448);
	}

	protected override void OnDisable()
	{
		_currentTween?.Kill();
		_background.Modulate = new Color(0.1f, 0.1f, 0.1f, 0.75f);
	}

	protected override void OnEnable()
	{
		_currentTween?.Kill();
		_background.Modulate = new Color(1f, 1f, 1f, 0f);
	}
}
