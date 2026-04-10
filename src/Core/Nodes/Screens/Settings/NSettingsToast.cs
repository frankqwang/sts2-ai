using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Localization;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

[Tool]
public partial class NSettingsToast : Control
{
	private MegaRichTextLabel _label;

	private Tween? _tween;

	private float _originalY;

	public override void _Ready()
	{
		_label = GetNode<MegaRichTextLabel>("Label");
		_originalY = base.Position.Y;
	}

	public void Show(LocString locString)
	{
		_label.SetTextAutoSize(locString.GetFormattedText());
		base.Scale = Vector2.One;
		Vector2 position = base.Position;
		position.Y = _originalY;
		base.Position = position;
		Color modulate = base.Modulate;
		modulate.A = 1f;
		base.Modulate = modulate;
		_tween?.Kill();
		_tween = CreateTween();
		_tween.TweenProperty(this, "position:y", _originalY - 120f, 0.25).SetTrans(Tween.TransitionType.Back).SetEase(Tween.EaseType.Out);
		_tween.TweenInterval(1.0);
		_tween.TweenProperty(this, "modulate:a", 0f, 0.5);
		_tween.TweenProperty(this, "scale", Vector2.One * 0.8f, 0.5);
	}
}
