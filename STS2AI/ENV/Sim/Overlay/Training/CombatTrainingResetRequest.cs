namespace MegaCrit.Sts2.Core.Training;

public sealed class CombatTrainingResetRequest
{
	public string? CharacterId { get; set; }

	public string? EncounterId { get; set; }

	public string? Seed { get; set; }

	public int? AscensionLevel { get; set; }
}
