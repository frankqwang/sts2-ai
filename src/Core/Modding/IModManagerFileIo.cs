using System.IO;
using Godot;

namespace MegaCrit.Sts2.Core.Modding;

public interface IModManagerFileIo
{
	string[] GetFilesAt(string path);

	string[] GetDirectoriesAt(string path);

	bool FileExists(string path);

	bool DirectoryExists(string path);

	Stream OpenStream(string path, Godot.FileAccess.ModeFlags mode);
}
