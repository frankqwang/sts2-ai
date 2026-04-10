using System;
using System.Text.RegularExpressions;
using Godot;
using MegaCrit.Sts2.Core.Logging;

namespace MegaCrit.Sts2.Core.Assets;

public partial class AtlasResourceLoader : ResourceFormatLoader
{
	private const string _atlasBasePath = "res://images/atlases/";

	private const string _spritesSuffix = ".sprites/";

	private static readonly StringName _typeAtlasTexture = new StringName("AtlasTexture");

	private static readonly StringName _typeTexture2D = new StringName("Texture2D");

	private static readonly StringName _typeResource = new StringName("Resource");

	private static readonly Regex _pathPattern = new Regex("^res://images/atlases/([^/]+)\\.sprites/(.+)\\.tres$", RegexOptions.Compiled);

	public override string[] _GetRecognizedExtensions()
	{
		return new string[1] { "tres" };
	}

	public override bool _HandlesType(StringName type)
	{
		if (!(type == _typeAtlasTexture) && !(type == _typeTexture2D))
		{
			return type == _typeResource;
		}
		return true;
	}

	public override string _GetResourceType(string path)
	{
		if (IsSpritePath(path))
		{
			return "AtlasTexture";
		}
		return "";
	}

	public override bool _RecognizePath(string path, StringName type)
	{
		return IsSpritePath(path);
	}

	public override bool _Exists(string path)
	{
		if (!IsSpritePath(path))
		{
			return false;
		}
		var (text, text2) = ParsePath(path);
		if (text == null || text2 == null)
		{
			return false;
		}
		if (!AtlasManager.IsAtlasLoaded(text))
		{
			AtlasManager.LoadAtlas(text);
		}
		if (!AtlasManager.HasSprite(text, text2))
		{
			return HasFallback(text, text2);
		}
		return true;
	}

	public override Variant _Load(string path, string originalPath, bool useSubThreads, int cacheMode)
	{
		if (!IsSpritePath(path))
		{
			return default(Variant);
		}
		var (text, text2) = ParsePath(path);
		if (text == null || text2 == null)
		{
			Log.Warn("AtlasResourceLoader: Failed to parse path: " + path);
			return 7L;
		}
		if (!AtlasManager.IsAtlasLoaded(text))
		{
			AtlasManager.LoadAtlas(text);
		}
		AtlasTexture sprite = AtlasManager.GetSprite(text, text2);
		if (sprite != null)
		{
			return sprite;
		}
		Texture2D texture2D = LoadFallback(text, text2);
		if (texture2D != null)
		{
			return texture2D;
		}
		if (!text2.StartsWith("mock_"))
		{
			Log.Warn($"AtlasResourceLoader: Missing sprite '{text2}' in {text} (requested: {path})");
		}
		return GetMissingTexture(text);
	}

	public override string[] _GetDependencies(string path, bool addTypes)
	{
		return Array.Empty<string>();
	}

	private static bool IsSpritePath(string path)
	{
		if (path.StartsWith("res://images/atlases/") && path.Contains(".sprites/"))
		{
			return path.EndsWith(".tres");
		}
		return false;
	}

	public static (string? AtlasName, string? SpriteName) ParsePath(string path)
	{
		Match match = _pathPattern.Match(path);
		if (!match.Success)
		{
			return (AtlasName: null, SpriteName: null);
		}
		return (AtlasName: match.Groups[1].Value, SpriteName: match.Groups[2].Value);
	}

	private static bool HasFallback(string atlasName, string spriteName)
	{
		string fallbackPath = GetFallbackPath(atlasName, spriteName);
		if (fallbackPath != null)
		{
			return ResourceLoader.Exists(fallbackPath);
		}
		return false;
	}

	private static Texture2D? LoadFallback(string atlasName, string spriteName)
	{
		string fallbackPath = GetFallbackPath(atlasName, spriteName);
		if (fallbackPath == null)
		{
			return null;
		}
		if (!ResourceLoader.Exists(fallbackPath))
		{
			return null;
		}
		Log.Debug($"AtlasResourceLoader: Using fallback for {atlasName}/{spriteName}: {fallbackPath}");
		return ResourceLoader.Load<Texture2D>(fallbackPath, null, ResourceLoader.CacheMode.Reuse);
	}

	private static string? GetFallbackPath(string atlasName, string spriteName)
	{
		switch (atlasName)
		{
		case "relic_atlas":
		case "relic_outline_atlas":
			return GetRelicFallbackPath(spriteName);
		case "power_atlas":
			return GetPowerFallbackPath(spriteName);
		case "card_atlas":
			return GetCardFallbackPath(spriteName);
		case "potion_atlas":
		case "potion_outline_atlas":
			return GetPotionFallbackPath(spriteName);
		default:
			return null;
		}
	}

	private static string? GetRelicFallbackPath(string spriteName)
	{
		string text = "res://images/relics/" + spriteName + ".png";
		if (ResourceLoader.Exists(text))
		{
			return text;
		}
		string text2 = "res://images/relics/beta/" + spriteName + ".png";
		if (ResourceLoader.Exists(text2))
		{
			return text2;
		}
		return null;
	}

	private static string? GetPowerFallbackPath(string spriteName)
	{
		string text = "res://images/powers/" + spriteName + ".png";
		if (ResourceLoader.Exists(text))
		{
			return text;
		}
		string text2 = "res://images/powers/beta/" + spriteName + ".png";
		if (ResourceLoader.Exists(text2))
		{
			return text2;
		}
		return null;
	}

	private static string? GetCardFallbackPath(string spriteName)
	{
		string text = "res://images/packed/card_portraits/" + spriteName + ".png";
		if (ResourceLoader.Exists(text))
		{
			return text;
		}
		int num = spriteName.LastIndexOf('/');
		if (num > 0)
		{
			string value = spriteName.Substring(0, num);
			int num2 = num + 1;
			string value2 = spriteName.Substring(num2, spriteName.Length - num2);
			string text2 = $"res://images/packed/card_portraits/{value}/beta/{value2}.png";
			if (ResourceLoader.Exists(text2))
			{
				return text2;
			}
		}
		return null;
	}

	private static string? GetPotionFallbackPath(string spriteName)
	{
		string text = "res://images/potions/" + spriteName + ".png";
		if (ResourceLoader.Exists(text))
		{
			return text;
		}
		return null;
	}

	private static Variant GetMissingTexture(string atlasName)
	{
		string text = ((!(atlasName == "card_atlas")) ? "res://images/powers/missing_power.png" : "res://images/packed/card_portraits/beta.png");
		string path = text;
		if (ResourceLoader.Exists(path))
		{
			Texture2D texture2D = ResourceLoader.Load<Texture2D>(path, null, ResourceLoader.CacheMode.Reuse);
			if (texture2D != null)
			{
				return texture2D;
			}
		}
		return 7L;
	}
}
