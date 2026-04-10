using Godot;

namespace MegaCrit.Sts2.Core.Nodes.Reaction;

public partial class NReactionWheelWedge : TextureRect
{
	private static readonly Color _defaultColor = new Color("e0f9ff40");

	private static readonly Color _selectedColor = new Color("c2f3ffc0");

	private TextureRect _textureRect;

	private Vector2 _normal;

	private Tween? _tween;

	private Vector2 _defaultPosition;

	public Texture2D Reaction => _textureRect.Texture;

	public override void _Ready()
	{
		_textureRect = GetNode<TextureRect>("TextureRect");
		_defaultPosition = base.Position;
	}

	public void OnSelected()
	{
		Vector2 vector = Vector2.Right.Rotated(base.Rotation);
		_tween?.Kill();
		_tween = CreateTween();
		_tween.SetParallel();
		_tween.TweenProperty(this, "position", _defaultPosition + vector * 25f, 0.05).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(this, "self_modulate", _selectedColor, 0.05).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	public void OnDeselected()
	{
		_tween?.Kill();
		_tween = CreateTween();
		_tween.SetParallel();
		_tween.TweenProperty(this, "position", _defaultPosition, 0.2).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(this, "self_modulate", _defaultColor, 0.2).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}
}
