using System.Collections.Generic;
using System.Text;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Modding;

namespace MegaCrit.Sts2.Core.Nodes.Screens.ModdingScreen;

public partial class NModInfoContainer : Control
{
	private MegaRichTextLabel _title;

	private TextureRect _image;

	private MegaRichTextLabel _description;

	public override void _Ready()
	{
		_title = GetNode<MegaRichTextLabel>("ModTitle");
		_image = GetNode<TextureRect>("ModImage");
		_description = GetNode<MegaRichTextLabel>("ModDescription");
		_title.Text = "";
		_image.Texture = null;
		_description.Text = "";
	}

	public void Fill(Mod mod)
	{
		_title.Text = mod.manifest?.name ?? "<No Name>";
		string path = "res://" + mod.manifest?.id + "/mod_image.png";
		if (ResourceLoader.Exists(path))
		{
			_image.Texture = PreloadManager.Cache.GetAsset<Texture2D>(path);
		}
		else
		{
			_image.Texture = null;
		}
		StringBuilder stringBuilder = new StringBuilder();
		StringBuilder stringBuilder2 = stringBuilder;
		StringBuilder stringBuilder3 = stringBuilder2;
		StringBuilder.AppendInterpolatedStringHandler handler = new StringBuilder.AppendInterpolatedStringHandler(21, 1, stringBuilder2);
		handler.AppendLiteral("[gold]Author[/gold]: ");
		handler.AppendFormatted(mod.manifest?.author ?? "unknown");
		stringBuilder3.AppendLine(ref handler);
		stringBuilder2 = stringBuilder;
		StringBuilder stringBuilder4 = stringBuilder2;
		handler = new StringBuilder.AppendInterpolatedStringHandler(22, 1, stringBuilder2);
		handler.AppendLiteral("[gold]Version[/gold]: ");
		handler.AppendFormatted(mod.manifest?.version ?? "unknown");
		stringBuilder4.AppendLine(ref handler);
		stringBuilder.AppendLine();
		stringBuilder2 = stringBuilder;
		StringBuilder stringBuilder5 = stringBuilder2;
		handler = new StringBuilder.AppendInterpolatedStringHandler(0, 1, stringBuilder2);
		handler.AppendFormatted(mod.manifest?.description ?? "No description");
		stringBuilder5.AppendLine(ref handler);
		List<LocString>? errors = mod.errors;
		if (errors != null && errors.Count > 0)
		{
			stringBuilder.AppendLine();
			foreach (LocString error in mod.errors)
			{
				stringBuilder2 = stringBuilder;
				StringBuilder stringBuilder6 = stringBuilder2;
				handler = new StringBuilder.AppendInterpolatedStringHandler(11, 1, stringBuilder2);
				handler.AppendLiteral("[red]");
				handler.AppendFormatted(error.GetFormattedText());
				handler.AppendLiteral("[/red]");
				stringBuilder6.AppendLine(ref handler);
			}
		}
		_description.Text = stringBuilder.ToString();
	}
}
