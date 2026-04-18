using System;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Platform.Steam;

public class SteamRemoteSaveStore : ICloudSaveStore
{
    public string? ReadFile(string path) => null;
    public Task<string?> ReadFileAsync(string path) => Task.FromResult<string?>(null);
    public void WriteFile(string path, string content) { }
    public void WriteFile(string path, byte[] content) { }
    public Task WriteFileAsync(string path, string content) => Task.CompletedTask;
    public Task WriteFileAsync(string path, byte[] content) => Task.CompletedTask;
    public bool FileExists(string path) => false;
    public bool DirectoryExists(string path) => false;
    public void DeleteFile(string path) { }
    public void RenameFile(string sourcePath, string destinationPath) { }
    public string[] GetFilesInDirectory(string directoryPath) => Array.Empty<string>();
    public string[] GetDirectoriesInDirectory(string directoryPath) => Array.Empty<string>();
    public void CreateDirectory(string directoryPath) { }
    public void DeleteDirectory(string directoryPath) { }
    public void DeleteTemporaryFiles(string directoryPath) { }
    public DateTimeOffset GetLastModifiedTime(string path) => DateTimeOffset.UnixEpoch;
    public int GetFileSize(string path) => 0;
    public void SetLastModifiedTime(string path, DateTimeOffset time) { }
    public string GetFullPath(string filename) => filename;
    public bool HasCloudFiles() => false;
    public void ForgetFile(string path) { }
    public bool IsFilePersisted(string path) => false;
    public void BeginSaveBatch() { }
    public void EndSaveBatch() { }
}
