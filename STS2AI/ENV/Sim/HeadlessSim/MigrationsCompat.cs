using System;
using System.Diagnostics.CodeAnalysis;
using MegaCrit.Sts2.SourceGeneration;

namespace MegaCrit.Sts2.Core.Saves.Migrations
{
	[GenerateSubtypes(DynamicallyAccessedMemberTypes = DynamicallyAccessedMemberTypes.PublicParameterlessConstructor)]
	public interface IMigration
	{
		int FromVersion { get; }

		int ToVersion { get; }

		Type SaveType { get; }

		MigratingData Migrate(MigratingData saveData);
	}

	public partial interface IMigration<T> : IMigration where T : ISaveSchema { }
}

namespace MegaCrit.Sts2.Core.Saves.Migrations.PrefsSaves
{
	public sealed class PrefsSaveV1ToV2 { }
}

namespace MegaCrit.Sts2.Core.Saves.Migrations.ProfileSaves
{
	public sealed class ProfileSaveV1ToV2 { }
}

namespace MegaCrit.Sts2.Core.Saves.Migrations.ProgressSaves
{
	public sealed class ProgressSaveV20ToV21 { }
}

namespace MegaCrit.Sts2.Core.Saves.Migrations.RunHistories
{
	public sealed class RunHistoryV7ToV8 { }
}

namespace MegaCrit.Sts2.Core.Saves.Migrations.SerializableRuns
{
	public sealed class SerializableRunV12ToV13 { }

	public sealed class SerializableRunV13ToV14 { }
}

namespace MegaCrit.Sts2.Core.Saves.Migrations.SettingsSaves
{
	public sealed class SettingsSaveV3ToV4 { }
}
