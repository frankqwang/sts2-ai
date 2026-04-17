using System.Text.Json.Serialization;
using MegaCrit.Sts2.Core.Models.Badges;

namespace MegaCrit.Sts2.Core.Saves;

public class BadgeStats
{
	[JsonPropertyName("id")]
	public required string Id { get; init; }

	[JsonPropertyName("count")]
	public required int Count { get; set; }

	[JsonPropertyName("rarity")]
	public required BadgeRarity Rarity { get; set; }
}
