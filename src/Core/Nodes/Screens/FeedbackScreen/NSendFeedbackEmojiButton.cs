using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.FeedbackScreen;

public partial class NSendFeedbackEmojiButton : NButton
{
	private static readonly Color _defaultColor = new Color(0.1f, 0.1f, 0.1f);

	private Tween? _tween;

	private bool _isSelected;

	private NSelectionReticle _selectionReticle;

	public override void _Ready()
	{
		ConnectSignals();
		_selectionReticle = GetNode<NSelectionReticle>("SelectionReticle");
		base.PivotOffset = base.Size * 0.5f;
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		base.Scale = Vector2.One * 1.2f;
		if (NControllerManager.Instance.IsUsingController)
		{
			_selectionReticle.OnSelect();
		}
	}

	protected override void OnUnfocus()
	{
		base.OnFocus();
		if (!_isSelected)
		{
			_selectionReticle.OnDeselect();
			base.Scale = Vector2.One;
		}
	}

	public void SetSelected(bool isSelected)
	{
		if (isSelected)
		{
			base.Scale = Vector2.One * 1.2f;
			base.SelfModulate = StsColors.blueGlow;
		}
		else
		{
			base.Scale = Vector2.One;
			base.SelfModulate = _defaultColor;
		}
		_isSelected = isSelected;
	}

	public void FlashError()
	{
		_tween?.Kill();
		_tween = CreateTween();
		_tween.TweenProperty(this, "self_modulate", StsColors.redGlow, 0.10000000149011612);
		_tween.TweenProperty(this, "self_modulate", _defaultColor, 0.10000000149011612);
		_tween.TweenProperty(this, "self_modulate", StsColors.redGlow, 0.10000000149011612);
		_tween.TweenProperty(this, "self_modulate", _defaultColor, 0.10000000149011612);
	}
}
