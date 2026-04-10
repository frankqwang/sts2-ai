using System;
using Godot;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Rooms;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NRemoteTargetingIndicator : Node2D
{
	private const int _segmentCount = 100;

	private const float _defaultAlpha = 0.5f;

	private const float _targetingAlpha = 1f;

	private Player _player;

	private Vector2 _fromPosition;

	private Vector2 _toPosition;

	private Line2D _line;

	private Line2D _lineBack;

	private Tween? _tween;

	private bool _isTargetingCreature;

	public override void _Ready()
	{
		_line = GetNode<Line2D>("Line");
		_lineBack = GetNode<Line2D>("LineBack");
		for (int i = 0; i < 101; i++)
		{
			_line.AddPoint(Vector2.Zero);
			_lineBack.AddPoint(Vector2.Zero);
		}
		StopDrawing();
	}

	public void Initialize(Player player)
	{
		_player = player;
		CharacterModel character = player.Character;
		_line.DefaultColor = character.RemoteTargetingLineColor;
		_lineBack.DefaultColor = character.RemoteTargetingLineOutline;
		Gradient gradient = _line.GetGradient();
		if (gradient != null)
		{
			for (int i = 0; i < gradient.GetPointCount(); i++)
			{
				gradient.SetColor(i, gradient.GetColor(i) * character.RemoteTargetingLineColor);
			}
			_line.SetGradient(gradient);
		}
		Gradient gradient2 = _lineBack.GetGradient();
		if (gradient2 != null)
		{
			for (int j = 0; j < gradient2.GetPointCount(); j++)
			{
				gradient2.SetColor(j, gradient2.GetColor(j) * character.RemoteTargetingLineOutline);
			}
			_lineBack.SetGradient(gradient);
		}
	}

	public override void _Process(double delta)
	{
		Vector2 zero = Vector2.Zero;
		zero.X = _fromPosition.X + (_toPosition.X - _fromPosition.X) * 0.5f;
		zero.Y = _fromPosition.Y - (_toPosition.Y - _fromPosition.Y) * 0.5f;
		for (int i = 0; i < 100; i++)
		{
			Vector2 position = MathHelper.BezierCurve(_fromPosition, _toPosition, zero, (float)i / 101f);
			_line.SetPointPosition(i, position);
			_lineBack.SetPointPosition(i, position);
		}
		_line.SetPointPosition(100, _toPosition);
		_lineBack.SetPointPosition(100, _toPosition);
		bool isTargetingCreature = false;
		foreach (Creature item in _player.Creature.CombatState?.Enemies ?? Array.Empty<Creature>())
		{
			NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(item);
			if (nCreature != null && nCreature.Hitbox.GetGlobalRect().HasPoint(base.GlobalPosition + _toPosition))
			{
				isTargetingCreature = true;
				break;
			}
		}
		DoTargetingCreatureTween(isTargetingCreature);
	}

	public void StartDrawingFrom(Vector2 from)
	{
		if (!NCombatUi.IsDebugHideMpTargetingUi)
		{
			_fromPosition = from;
			base.Visible = true;
			base.ProcessMode = (ProcessModeEnum)(base.Visible ? 0 : 4);
		}
	}

	public void StopDrawing()
	{
		base.Visible = false;
		base.ProcessMode = ProcessModeEnum.Disabled;
		Color modulate = base.Modulate;
		modulate.A = 0.5f;
		base.Modulate = modulate;
	}

	public void UpdateDrawingTo(Vector2 position)
	{
		_toPosition = position;
	}

	private void DoTargetingCreatureTween(bool isTargetingCreature)
	{
		if (isTargetingCreature != _isTargetingCreature)
		{
			_tween?.Kill();
			_tween = CreateTween();
			if (isTargetingCreature)
			{
				_tween.TweenProperty(this, "modulate:a", 1f, 0.10000000149011612);
			}
			else
			{
				_tween.TweenProperty(this, "modulate:a", 0.5f, 0.25);
			}
			_isTargetingCreature = isTargetingCreature;
		}
	}
}
