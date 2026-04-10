using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes;

public partial class NOneTimeInitialization : Node
{
	public override void _Ready()
	{
		OneTimeInitialization.Execute();
	}

	public bool ShouldReportSentryEvents()
	{
		if (SaveManager.Instance == null)
		{
			return false;
		}
		if (!SaveManager.Instance.PrefsSave.UploadData)
		{
			return false;
		}
		if (ModManager.LoadedMods.Count > 0)
		{
			return false;
		}
		return true;
	}
}
