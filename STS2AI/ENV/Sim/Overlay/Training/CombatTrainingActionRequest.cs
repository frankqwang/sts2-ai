namespace MegaCrit.Sts2.Core.Training;

public sealed class CombatTrainingActionRequest
{
	public CombatTrainingActionType Type { get; set; }

	public int? HandIndex { get; set; }

	public int? ChoiceIndex { get; set; }

	public int? Slot { get; set; }

	public uint? TargetId { get; set; }
}
