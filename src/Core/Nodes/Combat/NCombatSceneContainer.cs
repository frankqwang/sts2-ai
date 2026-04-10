using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NCombatSceneContainer : Control
{
	private const float _sixteenByNine = 1.7777778f;

	private const float _maxNarrowRatio = 1.3333334f;

	private Window _window;

	private Control _bgContainer;

	public override void _Ready()
	{
		_bgContainer = GetNode<Control>("%BgContainer");
		_window = GetTree().Root;
		_window.Connect(Viewport.SignalName.SizeChanged, Callable.From(OnWindowChange));
	}

	private void OnWindowChange()
	{
		float num = (float)_window.Size.X / (float)_window.Size.Y;
		if (num < 1.7777778f && SaveManager.Instance.SettingsSave.AspectRatioSetting == AspectRatioSetting.Auto)
		{
			_bgContainer.Scale = Vector2.One * Mathf.Max(MathHelper.Remap(num, 1.3333334f, 1.7777778f, 1.08f, 0.9f), 1.08f);
		}
		else
		{
			_bgContainer.Scale = Vector2.One * 0.9f;
		}
	}
}
