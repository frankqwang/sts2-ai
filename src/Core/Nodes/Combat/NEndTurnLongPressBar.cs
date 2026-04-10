using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NEndTurnLongPressBar : ColorRect
{
	private Control _outline;

	private double _pressTimer;

	private const double _longPressDuration = 0.5;

	private bool _isPressed;

	private const float _targetWidth = 204f;

	private NEndTurnButton _endTurnButton;

	private Tween? _tween;

	private bool _enabled = true;

	public override void _Ready()
	{
		_outline = GetNode<Control>("%BarOutline");
	}

	public void Init(NEndTurnButton endTurnButton)
	{
		_endTurnButton = endTurnButton;
	}

	public void StartPress()
	{
		_isPressed = true;
	}

	public void CancelPress()
	{
		_isPressed = false;
	}

	public override void _Process(double delta)
	{
		if (!_enabled)
		{
			return;
		}
		if (_isPressed)
		{
			_pressTimer += delta;
			if (_pressTimer > 0.5)
			{
				_enabled = false;
				base.Size = new Vector2(204f, 6f);
				_pressTimer = 0.0;
				_endTurnButton.CallReleaseLogic();
				TaskHelper.RunSafely(PlayAnim());
			}
			else
			{
				RecalculateBar();
			}
		}
		else if (_pressTimer > 0.0)
		{
			_pressTimer -= delta;
			if (_pressTimer < 0.0)
			{
				_pressTimer = 0.0;
				Color modulate = base.Modulate;
				modulate.A = 0f;
				base.Modulate = modulate;
			}
			else
			{
				RecalculateBar();
			}
		}
	}

	private void RecalculateBar()
	{
		float num = (float)(_pressTimer / 0.5);
		base.Size = new Vector2(num * 204f, 6f);
		base.Color = new Color(num * 2.5f, 0.6f + num, 0.6f);
		Color modulate = base.Modulate;
		modulate.A = Ease.CubicOut(num * 0.75f);
		base.Modulate = modulate;
	}

	private async Task PlayAnim()
	{
		_tween?.Kill();
		_tween = CreateTween();
		_tween.TweenProperty(this, "modulate:a", 0f, 0.25);
		await ToSignal(_tween, Tween.SignalName.Finished);
		base.Color = new Color(1f, 0.85f, 0.36f);
		_isPressed = false;
		_enabled = true;
	}
}
