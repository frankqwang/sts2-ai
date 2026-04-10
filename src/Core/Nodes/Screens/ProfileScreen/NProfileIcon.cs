using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.Nodes.Screens.ProfileScreen;

public partial class NProfileIcon : Control
{
	private TextureRect _icon;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlyArray<string>(new string[3]
	{
		ImageHelper.GetImagePath("ui/profile/profile_icon_1.png"),
		ImageHelper.GetImagePath("ui/profile/profile_icon_2.png"),
		ImageHelper.GetImagePath("ui/profile/profile_icon_3.png")
	});

	public override void _Ready()
	{
		_icon = GetNode<TextureRect>("Icon");
	}

	public void SetProfileId(int profileId)
	{
		_icon.Texture = ResourceLoader.Load<Texture2D>(ImageHelper.GetImagePath($"ui/profile/profile_icon_{profileId}.png"), null, ResourceLoader.CacheMode.Reuse);
	}
}
