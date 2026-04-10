using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Godot;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.Nodes.Debug;

public partial class NParticleCounter : Control
{
	private static readonly StringName _toggleParticleCounter = new StringName("toggle_particle_counter");

	private const float _secondsPerUpdate = 5f;

	private TextureRect _icon;

	private Label? _label;

	private double _secondsSinceLastUpdate;

	private int _totalParticles;

	private int _updateCount;

	public override void _Ready()
	{
		if (!OS.HasFeature("editor"))
		{
			this.QueueFreeSafely();
			return;
		}
		_icon = GetNode<TextureRect>("%Icon");
		_label = GetNode<Label>("%Label");
		base.Visible = false;
	}

	public override void _Input(InputEvent inputEvent)
	{
		CheckForHotkey(inputEvent);
	}

	private void CheckForHotkey(InputEvent inputEvent)
	{
		if (inputEvent.IsActionReleased(_toggleParticleCounter) && !NDevConsole.Instance.Visible)
		{
			base.Visible = !base.Visible;
		}
	}

	public override void _Process(double delta)
	{
		if (!base.Visible || _label == null)
		{
			return;
		}
		_secondsSinceLastUpdate += delta;
		if ((_totalParticles > 1 || _updateCount > 500) && _secondsSinceLastUpdate < 5.0)
		{
			return;
		}
		_updateCount++;
		_secondsSinceLastUpdate = 0.0;
		int num = 0;
		int num2 = 0;
		int num3 = 0;
		int num4 = 0;
		foreach (Node item in GetChildrenRecursive(GetTree().Root))
		{
			if (!(item is CpuParticles2D cpuParticles2D))
			{
				if (item is GpuParticles2D gpuParticles2D)
				{
					num3 += gpuParticles2D.Amount;
					num4++;
				}
			}
			else
			{
				num += cpuParticles2D.Amount;
				num2++;
			}
		}
		_totalParticles = num + num3;
		_label.Text = $"All particles: {_totalParticles}\nCPU particles: {num} in {num2} node{((num2 == 1) ? "" : "s")}\nGPU particles: {num3} in {num4} node{((num4 == 1) ? "" : "s")}";
	}

	private static List<Node> GetChildrenRecursive(Node root)
	{
		int num = 1;
		List<Node> list = new List<Node>(num);
		CollectionsMarshal.SetCount(list, num);
		Span<Node> span = CollectionsMarshal.AsSpan(list);
		int index = 0;
		span[index] = root;
		List<Node> list2 = list;
		foreach (Node child in root.GetChildren())
		{
			list2.AddRange(GetChildrenRecursive(child));
		}
		return list2;
	}
}
