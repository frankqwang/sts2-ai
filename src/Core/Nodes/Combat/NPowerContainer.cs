using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NPowerContainer : Control
{
	private Creature? _creature;

	private Vector2? _originalPosition;

	private readonly List<NPower> _powerNodes = new List<NPower>();

	public override void _EnterTree()
	{
		base._EnterTree();
		ConnectCreatureSignals();
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		if (_creature != null)
		{
			_creature.PowerApplied -= OnPowerApplied;
			_creature.PowerRemoved -= OnPowerRemoved;
		}
	}

	private void ConnectCreatureSignals()
	{
		if (_creature != null)
		{
			_creature.PowerApplied -= OnPowerApplied;
			_creature.PowerRemoved -= OnPowerRemoved;
			_creature.PowerApplied += OnPowerApplied;
			_creature.PowerRemoved += OnPowerRemoved;
		}
	}

	public void SetCreatureBounds(Control bounds)
	{
		base.GlobalPosition = new Vector2(bounds.GlobalPosition.X, base.GlobalPosition.Y);
		base.Size = new Vector2(bounds.Size.X * bounds.Scale.X + 25f, base.Size.Y);
		_originalPosition = base.Position;
		UpdatePositions();
	}

	private void Add(PowerModel power)
	{
		if (power.IsVisible)
		{
			NPower nPower = NPower.Create(power);
			nPower.Container = this;
			_powerNodes.Add(nPower);
			this.AddChildSafely(nPower);
			UpdatePositions();
		}
	}

	private void Remove(PowerModel power)
	{
		if (CombatManager.Instance.IsInProgress)
		{
			NPower nPower = _powerNodes.FirstOrDefault((NPower n) => n.Model == power);
			if (nPower != null)
			{
				_powerNodes.Remove(nPower);
				UpdatePositions();
				nPower.QueueFreeSafely();
			}
		}
	}

	private void UpdatePositions()
	{
		if (_powerNodes.Count != 0)
		{
			float x = _powerNodes[0].Size.X;
			float b = Mathf.CeilToInt(base.Size.X / x);
			b = Mathf.Max(Mathf.CeilToInt((float)_powerNodes.Count / 2f), b);
			for (int i = 0; i < _powerNodes.Count; i++)
			{
				_powerNodes[i].Position = new Vector2(x * ((float)i % b), Mathf.Floor((float)i / b) * x);
			}
			float num = x * Mathf.Min(b, _powerNodes.Count);
			base.Position = (_originalPosition ?? base.Position) + Vector2.Left * Mathf.Max(0f, num - base.Size.X) / 2f;
		}
	}

	public void SetCreature(Creature creature)
	{
		if (_creature != null)
		{
			throw new InvalidOperationException("Creature was already set.");
		}
		_creature = creature;
		ConnectCreatureSignals();
		foreach (PowerModel power in _creature.Powers)
		{
			Add(power);
		}
	}

	private void OnPowerApplied(PowerModel power)
	{
		Add(power);
	}

	private void OnPowerRemoved(PowerModel power)
	{
		Remove(power);
	}
}
