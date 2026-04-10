using Godot;

namespace MegaCrit.Sts2.Core.Nodes.Screens;

public partial class NAncientBgContainer : Control
{
	private Window _window;

	private const float _ratioMin = 1.3333f;

	private const float _ratioNormal = 1.7777f;

	private const float _ratioMax = 2.3333f;

	private Vector2 pos4_3 = new Vector2(-140f, 110f);

	private Vector2 scale4_3 = new Vector2(1f, 1f);

	private Vector2 pos16_9 = new Vector2(0f, 40f);

	private Vector2 scale16_9 = new Vector2(0.89f, 0.89f);

	private Vector2 pos21_9 = new Vector2(330f, 40f);

	private Vector2 scale21_9 = new Vector2(1f, 1f);

	public override void _Ready()
	{
		_window = GetTree().Root;
		_window.Connect(Viewport.SignalName.SizeChanged, Callable.From(OnWindowChange));
		OnWindowChange();
	}

	private void OnWindowChange()
	{
		float num = Mathf.Clamp(base.Size.X / base.Size.Y, 1.3333f, 2.3333f);
		base.PivotOffset = base.Size * 0.5f;
		if (num < 1.7777f)
		{
			float weight = Mathf.InverseLerp(1.3333f, 1.7777f, num);
			base.Position = pos4_3.Lerp(pos16_9, weight);
			base.Scale = scale4_3.Lerp(scale16_9, weight);
		}
		else
		{
			float weight2 = Mathf.InverseLerp(1.7777f, 2.3333f, num);
			base.Position = pos16_9.Lerp(pos21_9, weight2);
			base.Scale = scale16_9.Lerp(scale21_9, weight2);
		}
	}
}
