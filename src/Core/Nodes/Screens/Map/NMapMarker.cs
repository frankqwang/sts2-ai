using Godot;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Map;

public partial class NMapMarker : TextureRect
{
	private Tween? _tween;

	private Vector2 _posOffset;

	private bool _isEnabled;

	public override void _Ready()
	{
		_posOffset = new Vector2((0f - base.Size.X) / 2f, -35f);
	}

	public void Initialize(Player player)
	{
		_isEnabled = player.RunState.Players.Count == 1;
		base.Texture = player.Character.MapMarker;
		base.Visible = false;
	}

	public void ResetMapPoint()
	{
		base.Visible = false;
	}

	public void HideMapPoint()
	{
		if (_isEnabled)
		{
			_tween?.FastForwardToCompletion();
			_tween = CreateTween();
			_tween.TweenProperty(this, "scale", Vector2.Zero, 0.20000000298023224).From(Vector2.One);
			_tween.TweenCallback(Callable.From(delegate
			{
				base.Visible = false;
			}));
		}
	}

	public void SetMapPoint(NMapPoint node)
	{
		if (_isEnabled)
		{
			_tween?.FastForwardToCompletion();
			if (!base.Visible)
			{
				base.Visible = true;
				Vector2 vector = new Vector2(node.Size.X / 2f, 0f);
				base.Position = node.Position + vector + _posOffset;
				_tween = CreateTween();
				_tween.TweenProperty(this, "scale", Vector2.One, 0.20000000298023224).From(Vector2.Down);
				_tween.Parallel().TweenProperty(this, "position", base.Position + Vector2.Up * 25f, 0.20000000298023224).SetEase(Tween.EaseType.In)
					.SetTrans(Tween.TransitionType.Sine)
					.FromCurrent();
				_tween.TweenProperty(this, "position", base.Position, 0.75).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Elastic);
			}
		}
	}
}
