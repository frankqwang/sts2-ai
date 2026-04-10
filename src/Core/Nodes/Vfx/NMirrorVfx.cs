using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NMirrorVfx : Control
{
	private Sprite2D _mask1;

	private Control _reflection1;

	private Sprite2D _mask2;

	private Control _reflection2;

	private Sprite2D _mask3;

	private Control _reflection3;

	private FastNoiseLite _noise = new FastNoiseLite();

	private float _totalTime;

	private const float _noiseSpeed = 2f;

	private static string ScenePath => SceneHelper.GetScenePath("vfx/whole_screen/mirror_vfx");

	public static NMirrorVfx? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(ScenePath).Instantiate<NMirrorVfx>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		_mask1 = GetNode<Sprite2D>("Mask1");
		_reflection1 = GetNode<Control>("Mask1/Reflection");
		_mask2 = GetNode<Sprite2D>("Mask2");
		_reflection2 = GetNode<Control>("Mask2/Reflection");
		_mask3 = GetNode<Sprite2D>("Mask3");
		_reflection3 = GetNode<Control>("Mask3/Reflection");
	}

	public override void _Process(double delta)
	{
		_totalTime += (float)delta * 2f;
		_noise.Seed = 0;
		float num = 1.05f + _noise.GetNoise1D(_totalTime);
		_mask1.Scale = new Vector2(num, num);
		_reflection1.Scale = _mask1.Scale;
		_noise.Seed = 1;
		num = _noise.GetNoise1D(_totalTime);
		_mask1.RotationDegrees = Mathf.Abs(num) * 10f;
		_noise.Seed = 2;
		num = 1.05f + _noise.GetNoise1D(_totalTime);
		_mask2.Scale = new Vector2(num, num);
		_reflection2.Scale = new Vector2(num, num);
		_noise.Seed = 3;
		num = _noise.GetNoise1D(_totalTime);
		_mask2.RotationDegrees = Mathf.Abs(num) * 20f;
		_noise.Seed = 4;
		num = 1.05f + _noise.GetNoise1D(_totalTime);
		_mask3.Scale = new Vector2(num, num);
		_reflection3.Scale = new Vector2(num, num);
		_noise.Seed = 5;
		num = _noise.GetNoise1D(_totalTime);
		_mask3.RotationDegrees = Mathf.Abs(num) * 30f;
	}
}
