using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace MegaCrit.Sts2.Core.Modding;

public class ModManifest
{
	[JsonPropertyName("id")]
	public string? id;

	[JsonPropertyName("name")]
	public string? name;

	[JsonPropertyName("author")]
	public string? author;

	[JsonPropertyName("description")]
	public string? description;

	[JsonPropertyName("version")]
	public string? version;

	[JsonPropertyName("has_pck")]
	public bool hasPck;

	[JsonPropertyName("has_dll")]
	public bool hasDll;

	[JsonPropertyName("dependencies")]
	public List<string>? dependencies;

	[JsonPropertyName("affects_gameplay")]
	public bool affectsGameplay = true;
}
