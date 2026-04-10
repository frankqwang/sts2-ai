using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Logging;

namespace MegaCrit.Sts2.Core.Nodes.Debug;

public partial class NMemoryMonitor : Node
{
	private bool _isMonitoring;

	public override void _EnterTree()
	{
	}

	public override void _ExitTree()
	{
		StopMemoryMonitoring();
	}

	private async Task StartMemoryMonitoring()
	{
		_isMonitoring = true;
		while (_isMonitoring)
		{
			PrintMemoryUsage();
			PrintLargestAssets();
			await Task.Delay(10000);
		}
	}

	private void StopMemoryMonitoring()
	{
		_isMonitoring = false;
	}

	private static void PrintMemoryUsage()
	{
		Log.Info($"GetStaticMemoryUsage={OS.GetStaticMemoryUsage()}");
		Log.Info($"GetStaticMemoryPeakUsage={OS.GetStaticMemoryPeakUsage()}");
	}

	private static string FormatMemoryUsage(long bytes)
	{
		string[] array = new string[5] { "B", "KB", "MB", "GB", "TB" };
		int num = 0;
		while (bytes >= 1024 && num < array.Length - 1)
		{
			num++;
			bytes /= (long)Math.Pow(1024.0, num);
		}
		return $"{bytes:0.##} {array[num]}";
	}

	private static void PrintLargestAssets()
	{
		var enumerable = (from assetPath in PreloadManager.Cache.GetCacheKeys()
			select new
			{
				Path = assetPath,
				Size = GetFileSizeInMb(assetPath)
			} into file
			orderby file.Size descending
			select file).Take(10);
		Log.Info("Top 10 largest files in asset cache:");
		foreach (var item in enumerable)
		{
			Log.Info($"Size: {item.Size:F3} MB, Path: {item.Path}");
		}
	}

	private static float GetFileSizeInMb(string godotPath)
	{
		string text = ProjectSettings.GlobalizePath(godotPath);
		if (File.Exists(text))
		{
			long length = new FileInfo(text).Length;
			return (float)length / 1048576f;
		}
		Log.Info("File does not exist: " + text);
		return 0f;
	}
}
