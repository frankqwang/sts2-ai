using System;
using System.Text.Json.Serialization;

namespace MegaCrit.Sts2.Core.Modding;

public class SettingsSaveMod
{
	[JsonPropertyName("id")]
	public string Id { get; set; } = "";

	[JsonPropertyName("source")]
	public ModSource Source { get; set; }

	[JsonPropertyName("is_enabled")]
	public bool IsEnabled { get; set; } = true;

	public SettingsSaveMod()
	{
	}

	public SettingsSaveMod(Mod mod)
	{
		Id = mod.manifest?.id ?? throw new InvalidOperationException();
		Source = mod.modSource;
	}
}
