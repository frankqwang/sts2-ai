// Generic IMigration<T> that MigrationBase<T> inherits from.
// Original is in MegaCrit/ directory (excluded). Defined here so the game code compiles.
namespace MegaCrit.Sts2.Core.Saves.Migrations;

public partial interface IMigration<T> : IMigration where T : ISaveSchema { }
