using Godot;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.Nodes.Debug;

public partial class NDebugAspectRatio : Control
{
	private Window _window;

	private Label _infoLabel;

	private TextureRect _bg;

	private const float _maxNarrowRatio = 1.3333334f;

	private const float _maxWideRatio = 2.3888888f;

	private static readonly Vector2 _defaultBgScale = Vector2.One * 1.01f;

	private const float _bgScaleRatioThreshold = 1.5f;

	public override void _Ready()
	{
		_bg = GetNode<TextureRect>("EventBg");
		_infoLabel = GetNode<Label>("Anchors/Label");
		_window = GetTree().Root;
		_window.Connect(Viewport.SignalName.SizeChanged, Callable.From(OnWindowChange));
	}

	private void OnWindowChange()
	{
		float num = (float)_window.Size.X / (float)_window.Size.Y;
		string value = num.ToString("0.000");
		ScaleBgIfNarrow(num);
		if (num > 2.3888888f)
		{
			_window.ContentScaleAspect = Window.ContentScaleAspectEnum.KeepWidth;
			_window.ContentScaleSize = new Vector2I(2580, 1080);
			_infoLabel.Text = $"{value}: {_window.Size}";
			_infoLabel.Modulate = StsColors.red;
		}
		else if (num < 1.3333334f)
		{
			_window.ContentScaleAspect = Window.ContentScaleAspectEnum.KeepHeight;
			_window.ContentScaleSize = new Vector2I(1680, 1260);
			_infoLabel.Text = $"{value}: {_window.Size}";
			_infoLabel.Modulate = StsColors.red;
		}
		else
		{
			_window.ContentScaleAspect = Window.ContentScaleAspectEnum.Expand;
			_window.ContentScaleSize = new Vector2I(1680, 1080);
			_infoLabel.Text = $"{value}: {_window.Size}";
			_infoLabel.Modulate = StsColors.cream;
		}
	}

	private void ScaleBgIfNarrow(float ratio)
	{
		if (ratio < 1.5f)
		{
			_bg.Scale = Vector2.One * 1.05f;
		}
		else
		{
			_bg.Scale = _defaultBgScale;
		}
	}
}
