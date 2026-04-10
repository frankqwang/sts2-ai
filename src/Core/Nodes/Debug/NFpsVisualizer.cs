using System.Globalization;
using Godot;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.Nodes.Debug;

public partial class NFpsVisualizer : TextureRect
{
	[Signal]
	public delegate void MouseReleasedEventHandler(InputEvent inputEvent);

	private Label? _label;

	[Export(PropertyHint.None, "")]
	private Texture2D _happy;

	[Export(PropertyHint.None, "")]
	private Texture2D _content;

	[Export(PropertyHint.None, "")]
	private Texture2D _neutral;

	[Export(PropertyHint.None, "")]
	private Texture2D _sad;

	public override void _Ready()
	{
		if (!OS.HasFeature("editor"))
		{
			this.QueueFreeSafely();
			return;
		}
		_label = GetNode<Label>("Label");
		Connect(SignalName.MouseReleased, Callable.From<InputEvent>(HandleMouseRelease));
	}

	private void HandleMouseRelease(InputEvent inputEvent)
	{
		this.QueueFreeSafely();
	}

	public override void _GuiInput(InputEvent inputEvent)
	{
		if (inputEvent is InputEventMouseButton inputEventMouseButton && inputEventMouseButton.ButtonIndex == MouseButton.Left && inputEventMouseButton.IsReleased())
		{
			EmitSignal(SignalName.MouseReleased, inputEvent);
		}
	}

	public override void _Process(double delta)
	{
		if (_label != null)
		{
			double framesPerSecond = Engine.GetFramesPerSecond();
			Texture2D texture = ((framesPerSecond >= 58.0) ? _happy : ((framesPerSecond >= 50.0) ? _content : ((!(framesPerSecond >= 30.0)) ? _sad : _neutral)));
			base.Texture = texture;
			_label.Text = framesPerSecond.ToString(CultureInfo.InvariantCulture);
		}
	}
}
