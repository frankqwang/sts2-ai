using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Saves.Migrations.Shared;

namespace MegaCrit.Sts2.Core.Saves.Migrations.SerializableRuns;

[Migration(typeof(SerializableRun), 15, 16)]
public class SerializableRunV15ToV16 : MigrationBase<SerializableRun>
{
	protected override void ApplyMigration(MigratingData saveData)
	{
		Log.Info("SerializableRun migration v15 -> v16: Migrating renamed/deleted ModelIds");
		SharedMigrationHelper.ReplaceModelIds(saveData.GetRawNode(), SharedMigrationHelper.V100Renames);
	}
}
