using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Debug;
using MegaCrit.Sts2.Core.Logging;

namespace MegaCrit.Sts2.Core.Saves;

public class CloudSaveStore : ICloudSaveStore, ISaveStore
{
	public ISaveStore LocalStore { get; }

	public ICloudSaveStore CloudStore { get; }

	public CloudSaveStore(ISaveStore localStore, ICloudSaveStore cloudStore)
	{
		LocalStore = localStore;
		CloudStore = cloudStore;
	}

	public string? ReadFile(string path)
	{
		return LocalStore.ReadFile(path);
	}

	public Task<string?> ReadFileAsync(string path)
	{
		return LocalStore.ReadFileAsync(path);
	}

	public bool FileExists(string path)
	{
		return LocalStore.FileExists(path);
	}

	public bool DirectoryExists(string path)
	{
		return LocalStore.DirectoryExists(path);
	}

	public void WriteFile(string path, string content)
	{
		LocalStore.WriteFile(path, content);
		try
		{
			CloudStore.WriteFile(path, content);
			SyncLocalTimestamp(path);
		}
		catch (Exception ex)
		{
			Log.Warn("Cloud write failed for " + path + ", local file preserved: " + ex.Message);
			SyncLocalTimestamp(path);
		}
	}

	public void WriteFile(string path, byte[] bytes)
	{
		LocalStore.WriteFile(path, bytes);
		try
		{
			CloudStore.WriteFile(path, bytes);
			SyncLocalTimestamp(path);
		}
		catch (Exception ex)
		{
			Log.Warn("Cloud write failed for " + path + ", local file preserved: " + ex.Message);
			SyncLocalTimestamp(path);
		}
	}

	public async Task WriteFileAsync(string path, string content)
	{
		await LocalStore.WriteFileAsync(path, content);
		try
		{
			await CloudStore.WriteFileAsync(path, content);
			SyncLocalTimestamp(path);
		}
		catch (Exception ex)
		{
			Log.Warn("Cloud write failed for " + path + ", local file preserved: " + ex.Message);
			SyncLocalTimestamp(path);
		}
	}

	public async Task WriteFileAsync(string path, byte[] bytes)
	{
		await LocalStore.WriteFileAsync(path, bytes);
		try
		{
			await CloudStore.WriteFileAsync(path, bytes);
			SyncLocalTimestamp(path);
		}
		catch (Exception ex)
		{
			Log.Warn("Cloud write failed for " + path + ", local file preserved: " + ex.Message);
			SyncLocalTimestamp(path);
		}
	}

	public void DeleteFile(string path)
	{
		LocalStore.DeleteFile(path);
		try
		{
			CloudStore.DeleteFile(path);
		}
		catch (Exception ex)
		{
			Log.Warn("Cloud delete failed for " + path + ", local delete preserved: " + ex.Message);
		}
	}

	public void RenameFile(string sourcePath, string destinationPath)
	{
		LocalStore.RenameFile(sourcePath, destinationPath);
		try
		{
			CloudStore.RenameFile(sourcePath, destinationPath);
		}
		catch (Exception ex)
		{
			Log.Warn($"Cloud rename failed for {sourcePath} -> {destinationPath}, local rename preserved: {ex.Message}");
		}
	}

	public string[] GetFilesInDirectory(string directoryPath)
	{
		return LocalStore.GetFilesInDirectory(directoryPath);
	}

	public string[] GetDirectoriesInDirectory(string directoryPath)
	{
		return LocalStore.GetDirectoriesInDirectory(directoryPath);
	}

	public void CreateDirectory(string directoryPath)
	{
		LocalStore.CreateDirectory(directoryPath);
		try
		{
			CloudStore.CreateDirectory(directoryPath);
		}
		catch (Exception ex)
		{
			Log.Warn("Cloud create directory failed for " + directoryPath + ": " + ex.Message);
		}
	}

	public void DeleteDirectory(string directoryPath)
	{
		LocalStore.DeleteDirectory(directoryPath);
		try
		{
			CloudStore.DeleteDirectory(directoryPath);
		}
		catch (Exception ex)
		{
			Log.Warn("Cloud delete directory failed for " + directoryPath + ": " + ex.Message);
		}
	}

	public void DeleteTemporaryFiles(string directoryPath)
	{
		LocalStore.DeleteTemporaryFiles(directoryPath);
		try
		{
			CloudStore.DeleteTemporaryFiles(directoryPath);
		}
		catch (Exception ex)
		{
			Log.Warn("Cloud delete temporary files failed for " + directoryPath + ": " + ex.Message);
		}
	}

	public DateTimeOffset GetLastModifiedTime(string path)
	{
		return LocalStore.GetLastModifiedTime(path);
	}

	public int GetFileSize(string path)
	{
		return LocalStore.GetFileSize(path);
	}

	public void SetLastModifiedTime(string path, DateTimeOffset time)
	{
		LocalStore.SetLastModifiedTime(path, time);
	}

	public string GetFullPath(string filename)
	{
		return LocalStore.GetFullPath(filename);
	}

	public bool HasCloudFiles()
	{
		return CloudStore.HasCloudFiles();
	}

	public void ForgetFile(string path)
	{
		try
		{
			CloudStore.ForgetFile(path);
		}
		catch (Exception ex)
		{
			Log.Warn("Cloud forget failed for " + path + ": " + ex.Message);
		}
	}

	public bool IsFilePersisted(string path)
	{
		return CloudStore.IsFilePersisted(path);
	}

	public void BeginSaveBatch()
	{
		CloudStore.BeginSaveBatch();
	}

	public void EndSaveBatch()
	{
		CloudStore.EndSaveBatch();
	}

	public async Task SyncCloudToLocal(string path)
	{
		try
		{
			await SyncCloudToLocalInternal(path);
		}
		catch (Exception ex)
		{
			Log.Warn("SteamRemoteStorage: Failed to sync " + path + " from cloud, skipping: " + ex.Message);
			SentryService.CaptureException(ex);
		}
	}

	private async Task SyncCloudToLocalInternal(string path)
	{
		bool flag = CloudStore.FileExists(path);
		bool flag2 = LocalStore.FileExists(path);
		if (flag)
		{
			DateTimeOffset lastModifiedTime = CloudStore.GetLastModifiedTime(path);
			DateTimeOffset? dateTimeOffset = (flag2 ? new DateTimeOffset?(LocalStore.GetLastModifiedTime(path)) : ((DateTimeOffset?)null));
			if (!flag2 || lastModifiedTime != dateTimeOffset)
			{
				Log.Info($"Copying {path} from cloud to local. Local file exists: {flag2} Cloud save time: {lastModifiedTime} Local save time: {dateTimeOffset}");
				string content = await CloudStore.ReadFileAsync(path);
				await LocalStore.WriteFileAsync(path, content);
				SyncLocalTimestamp(path);
			}
			else
			{
				Log.Debug($"Skipping sync for {path}, last modified time matches on local and remote ({lastModifiedTime})");
			}
		}
		else if (flag2)
		{
			Log.Info("Deleting " + path + " because it does not exist on remote");
			LocalStore.DeleteFile(path);
			LocalStore.DeleteFile(path + ".backup");
		}
		else
		{
			Log.Debug("Skipping sync for " + path + ", it doesn't exist on either local or cloud");
		}
	}

	public IEnumerable<Task> SyncCloudToLocalDirectory(string directoryPath)
	{
		Log.Debug("Syncing all files in " + directoryPath + " from cloud to local");
		HashSet<string> filePathsRead = new HashSet<string>();
		string[] array = Array.Empty<string>();
		try
		{
			if (CloudStore.DirectoryExists(directoryPath))
			{
				array = CloudStore.GetFilesInDirectory(directoryPath);
			}
		}
		catch (Exception ex)
		{
			Log.Warn("Failed to list cloud files in " + directoryPath + ", skipping cloud sync: " + ex.Message);
			SentryService.CaptureException(ex);
		}
		string[] array2 = array;
		foreach (string text in array2)
		{
			string text2 = directoryPath + "/" + text;
			filePathsRead.Add(text2);
			Log.Debug("Checking file " + text2 + " in cloud saves");
			yield return SyncCloudToLocal(text2);
		}
		if (!LocalStore.DirectoryExists(directoryPath))
		{
			yield break;
		}
		array2 = LocalStore.GetFilesInDirectory(directoryPath);
		foreach (string text3 in array2)
		{
			string text4 = directoryPath + "/" + text3;
			if (!filePathsRead.Contains(text4))
			{
				Log.Debug("Checking file " + text4 + " in local saves");
				yield return SyncCloudToLocal(text4);
			}
		}
	}

	public async Task OverwriteCloudWithLocal(string path, bool forgetImmediately = false)
	{
		if (LocalStore.FileExists(path))
		{
			Log.Debug("Writing file " + path + " to cloud");
			string content = await LocalStore.ReadFileAsync(path);
			try
			{
				await CloudStore.WriteFileAsync(path, content);
				if (forgetImmediately)
				{
					try
					{
						Log.Debug("Immediately forgetting " + path);
						CloudStore.ForgetFile(path);
					}
					catch (Exception ex)
					{
						Log.Warn("Cloud forget failed for " + path + ": " + ex.Message);
					}
				}
				SyncLocalTimestamp(path);
				return;
			}
			catch (Exception ex2)
			{
				Log.Warn("Cloud write failed for " + path + ", local file preserved: " + ex2.Message);
				SyncLocalTimestamp(path);
				return;
			}
		}
		try
		{
			if (CloudStore.FileExists(path))
			{
				Log.Debug("Deleting file " + path + " from cloud because it doesn't exist on local");
				CloudStore.DeleteFile(path);
			}
		}
		catch (Exception ex3)
		{
			Log.Warn("Cloud delete failed for " + path + ": " + ex3.Message);
		}
	}

	public IEnumerable<Task> OverwriteCloudWithLocalDirectory(string directoryPath, int? byteLimit, int? fileLimit)
	{
		Log.Debug("Writing all files in directory " + directoryPath + " to cloud");
		HashSet<string> filePathsRead = new HashSet<string>();
		string[] array = Array.Empty<string>();
		try
		{
			if (CloudStore.DirectoryExists(directoryPath))
			{
				array = CloudStore.GetFilesInDirectory(directoryPath);
			}
		}
		catch (Exception ex)
		{
			Log.Warn("Failed to list cloud files in " + directoryPath + ", skipping cloud delete sync: " + ex.Message);
			SentryService.CaptureException(ex);
		}
		string[] array2 = array;
		foreach (string text in array2)
		{
			filePathsRead.Add(text);
			yield return OverwriteCloudWithLocal(directoryPath + "/" + text);
		}
		if (!LocalStore.DirectoryExists(directoryPath))
		{
			yield break;
		}
		List<string> list = LocalStore.GetFilesInDirectory(directoryPath).ToList();
		int i = 0;
		int totalFilesWritten = 0;
		if (byteLimit.HasValue || fileLimit.HasValue)
		{
			list.Sort((string p1, string p2) => LocalStore.GetLastModifiedTime(directoryPath + "/" + p2).CompareTo(LocalStore.GetLastModifiedTime(directoryPath + "/" + p1)));
		}
		foreach (string item in list)
		{
			if (!filePathsRead.Contains(item) && !item.EndsWith(".backup"))
			{
				string path = directoryPath + "/" + item;
				int bytesToWrite = LocalStore.GetFileSize(path);
				bool flag = (byteLimit.HasValue && i + bytesToWrite > byteLimit.Value) || (fileLimit.HasValue && totalFilesWritten + 1 > fileLimit.Value);
				if (flag)
				{
					Log.Info($"File {item} will be immediately forgotten after writing to cloud. Bytes written:{i + bytesToWrite}. Files written: {totalFilesWritten + 1}");
				}
				yield return OverwriteCloudWithLocal(path, flag);
				i += bytesToWrite;
				totalFilesWritten++;
			}
		}
	}

	public void ForgetFilesInDirectoryBeforeWritingIfNecessary(string directoryPath, int bytesToBeWritten, int byteLimit, int fileLimit)
	{
		try
		{
			ForgetFilesInDirectoryBeforeWritingIfNecessaryInternal(directoryPath, bytesToBeWritten, byteLimit, fileLimit);
		}
		catch (Exception ex)
		{
			Log.Warn("Cloud quota management failed for " + directoryPath + ": " + ex.Message);
			SentryService.CaptureException(ex);
		}
	}

	private void ForgetFilesInDirectoryBeforeWritingIfNecessaryInternal(string directoryPath, int bytesToBeWritten, int byteLimit, int fileLimit)
	{
		int num = bytesToBeWritten;
		int num2 = 1;
		string[] filesInDirectory = CloudStore.GetFilesInDirectory(directoryPath);
		List<string> list = new List<string>();
		string[] array = filesInDirectory;
		foreach (string text in array)
		{
			string text2 = directoryPath + "/" + text;
			if (CloudStore.IsFilePersisted(text2))
			{
				list.Add(text2);
				num += CloudStore.GetFileSize(text2);
				num2++;
			}
		}
		if (num > byteLimit || num2 > fileLimit)
		{
			list.Sort((string p1, string p2) => GetLastModifiedTime(p2).CompareTo(GetLastModifiedTime(p1)));
			while (num > byteLimit || num2 > fileLimit)
			{
				string text3 = list[list.Count - 1];
				num -= CloudStore.GetFileSize(text3);
				num2--;
				Log.Info($"Forgetting file {text3} from cloud storage because we're past our quota. Bytes after forgetting: {num}. Files after forgetting: {num2}");
				CloudStore.ForgetFile(text3);
				list.RemoveAt(list.Count - 1);
			}
		}
	}

	private void SyncLocalTimestamp(string path)
	{
		try
		{
			LocalStore.SetLastModifiedTime(path, CloudStore.GetLastModifiedTime(path));
		}
		catch (Exception ex)
		{
			Log.Warn("Failed to sync timestamp for " + path + ", will re-sync on next launch: " + ex.Message);
		}
	}
}
