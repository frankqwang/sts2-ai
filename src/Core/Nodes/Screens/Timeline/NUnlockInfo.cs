using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Timeline;

public partial class NUnlockInfo : Control
{
	private TextureRect _icon;

	private MegaRichTextLabel _label;

	private Tween? _tween;

	public override void _Ready()
	{
		_icon = GetNode<TextureRect>("%Icon");
		_label = GetNode<MegaRichTextLabel>("%Text");
	}

	public void HideImmediately()
	{
		Color modulate = base.Modulate;
		modulate.A = 0f;
		base.Modulate = modulate;
	}

	public void AnimIn(string text)
	{
		Color modulate = base.Modulate;
		modulate.A = 0f;
		base.Modulate = modulate;
		_label.Text = text;
		_tween?.Kill();
		_tween = CreateTween();
		_tween.TweenProperty(this, "modulate:a", 0.8f, 1.0);
	}

	public async Task AnimInViaPaginator(string text)
	{
		_tween?.Kill();
		_tween = CreateTween();
		_tween.TweenProperty(this, "modulate:a", 0f, 0.1);
		await ToSignal(_tween, Tween.SignalName.Finished);
		_label.Text = text;
		_tween?.Kill();
		_tween = CreateTween();
		_tween.TweenProperty(this, "modulate:a", 0.8f, 1.0);
	}
}
