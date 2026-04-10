using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NSettingsGradientMask : TextureRect
{
	private const float _fadeOffset = -8f;

	private const float _fadeSize = 16f;

	private NSettingsTabManager _tabContainer;

	private GradientTexture2D _texture;

	public override void _Ready()
	{
		NSettingsScreen ancestorOfType = this.GetAncestorOfType<NSettingsScreen>();
		_tabContainer = ancestorOfType.GetNode<NSettingsTabManager>("%SettingsTabManager");
		_texture = (GradientTexture2D)base.Texture;
		Connect(Control.SignalName.Resized, Callable.From(OnResized));
		OnResized();
	}

	private void OnResized()
	{
		float num = 1f - (_tabContainer.Position.Y + _tabContainer.Size.Y + -8f + 16f) / base.Size.Y;
		float offset = num + 16f / base.Size.Y;
		_texture.Gradient.SetOffset(2, num);
		_texture.Gradient.SetOffset(3, offset);
		_texture.Gradient.SetColor(2, Colors.White);
		_texture.Gradient.SetColor(3, StsColors.transparentWhite);
	}
}
