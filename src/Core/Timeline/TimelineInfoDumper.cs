using System;
using System.Collections.Generic;
using System.Linq;
using Godot;

namespace MegaCrit.Sts2.Core.Timeline;

public partial class TimelineInfoDumper : Node
{
	public static void Dump()
	{
		List<EpochModel> allEpochs = GetAllEpochs();
		Console.Out.WriteLine("START TIMELINE INFO DUMPER");
		Console.Out.WriteLine("START TIMELINE INFO DUMPER");
		Console.Out.WriteLine("START TIMELINE INFO DUMPER");
		foreach (EpochModel item in allEpochs)
		{
			Console.Out.WriteLine($"\"{item.Id}\", \"{item.Era}\", \"{(int)item.Era}.{item.EraPosition}\", \"{item.Title}\", \"{item.Description.Replace("\r", "").Replace("\n", "")}\", \"{item.UnlockText}\", \"{item.BigPortraitPath}\"");
		}
		Console.Out.WriteLine("END TIMELINE INFO DUMPER");
		Console.Out.WriteLine("END TIMELINE INFO DUMPER");
		Console.Out.WriteLine("END TIMELINE INFO DUMPER");
	}

	public static List<EpochModel> GetAllEpochs()
	{
		return EpochModel.AllEpochIds.Select(EpochModel.Get).ToList();
	}
}
