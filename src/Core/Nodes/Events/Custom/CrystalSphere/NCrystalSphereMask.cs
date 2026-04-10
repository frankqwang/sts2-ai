using Godot;
using Godot.Collections;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;

namespace MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;

public partial class NCrystalSphereMask : Control
{
	private static readonly StringName _gridFadeParams = new StringName("gridFadeParams");

	private static readonly StringName _timeStr = new StringName("time");

	private ShaderMaterial _material;

	private Array<Vector3> _values = new Array<Vector3>();

	private float _time;

	public override void _Ready()
	{
		_material = (ShaderMaterial)base.Material;
		for (int i = 0; i < 121; i++)
		{
			_values.Add(Vector3.Zero);
		}
		_values[3] = new Vector3(1f, 0f, 0f);
	}

	public override void _Process(double delta)
	{
		_time += (float)delta;
		_material.SetShaderParameter(_timeStr, _time);
	}

	public void UpdateMat(CrystalSphereCell cell)
	{
		int index = cell.Y * 11 + cell.X;
		float x = _values[index].Y;
		if (_values[index].Z == 0f)
		{
			x = 1f;
		}
		_values[index] = new Vector3(x, cell.IsHidden ? 1 : 0, _time);
		_material.SetShaderParameter(_gridFadeParams, _values);
	}
}
