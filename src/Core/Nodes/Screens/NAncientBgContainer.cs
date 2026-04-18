using Godot;

namespace MegaCrit.Sts2.Core.Nodes.Screens;

public partial class NAncientBgContainer : Control
{
	private Window _window;

	private const float _ratioMin = 1.3333f;

	private const float _ratioNormal = 1.7777f;

	private const float _ratioMax = 2.3333f;

	private Vector2 _pos43 = new Vector2(-140f, 110f);

	private Vector2 _scale43 = new Vector2(1f, 1f);

	private Vector2 _pos169 = new Vector2(0f, 40f);

	private Vector2 _scale169 = new Vector2(0.89f, 0.89f);

	private Vector2 _pos219 = new Vector2(330f, 40f);

	private Vector2 _scale219 = new Vector2(1f, 1f);

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
			base.Position = _pos43.Lerp(_pos169, weight);
			base.Scale = _scale43.Lerp(_scale169, weight);
		}
		else
		{
			float weight2 = Mathf.InverseLerp(1.7777f, 2.3333f, num);
			base.Position = _pos169.Lerp(_pos219, weight2);
			base.Scale = _scale169.Lerp(_scale219, weight2);
		}
	}
}
