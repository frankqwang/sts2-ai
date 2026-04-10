using System;
using System.Linq;

namespace Godot;

// === Engine, OS, Time, GD — the "global" Godot singletons ===

public static class Engine
{
    public static double TimeScale { get; set; } = 1.0;
    public static int PhysicsTicksPerSecond { get; set; } = 60;
    public static int MaxFps { get; set; } = 0;
    public static bool IsEditorHint() => false;
    // CRITICAL: returning null triggers all "headless" guards in simulator code
    public static SceneTree? GetMainLoop() => null;
    public static int GetPhysicsFrames() => 0;
    public static int GetProcessFrames() => 0;
    public static double GetPhysicsInterpolationFraction() => 0;
    public static bool HasSingleton(StringName name) => false;
    public static GodotObject? GetSingleton(StringName name) => null;
}

public static partial class OS
{
    public static string[] GetCmdlineArgs() => Environment.GetCommandLineArgs()[1..];
    public static string[] GetCmdlineUserArgs() => GetCmdlineArgs();
    public static bool HasFeature(string feature) => feature switch
    {
        "editor" => false,
        "standalone" => true,
        "debug" => true,
        _ => false,
    };
    public static string GetUniqueId() => "headless-sim";
    public static int GetProcessorCount() => Environment.ProcessorCount;
    public static string GetUserDataDir() => AppDomain.CurrentDomain.BaseDirectory;
    public static string GetExecutablePath() => Environment.ProcessPath ?? "";
    public static string GetName() => "HeadlessSim";
    public static int GetProcessId() => Environment.ProcessId;
    public static ulong GetStaticMemoryUsage() => (ulong)GC.GetTotalMemory(false);

    public static bool IsDebugBuild() => true;
    public static System.Collections.Generic.Dictionary<string, ulong> GetMemoryInfo() => new();
}

public static class Time
{
    private static readonly long _startTicks = Environment.TickCount64;
    public static ulong GetTicksMsec() => (ulong)(Environment.TickCount64 - _startTicks);
    public static ulong GetTicksUsec() => GetTicksMsec() * 1000;
    public static double GetUnixTimeFromSystem() => DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0;
}

public static class GD
{
    public static void Print(params object[] args) => Console.Error.WriteLine(string.Join("", args));
    public static void PrintErr(params object[] args) => Console.Error.WriteLine("[ERR] " + string.Join("", args));
    public static void PrintRich(params object[] args) => Console.Error.WriteLine(string.Join("", args));
    public static void PushError(params object[] args) => Console.Error.WriteLine("[ERR] " + string.Join("", args));
    public static void PushWarning(params object[] args) => Console.Error.WriteLine("[WARN] " + string.Join("", args));
    public static float Randf() => Random.Shared.NextSingle();
    public static int RandRange(int from, int to) => Random.Shared.Next(from, to + 1);
    public static float RandRange(float from, float to) => from + Random.Shared.NextSingle() * (to - from);
    public static Variant BytesToVar(byte[] bytes) => default;
    public static byte[] VarToBytes(Variant v) => Array.Empty<byte>();
    public static Variant StrToVar(string s) => default;
    public static string VarToStr(Variant v) => "";
}

public static partial class ProjectSettings
{
    public static string GlobalizePath(string path) => path.Replace("res://", AppDomain.CurrentDomain.BaseDirectory).Replace("user://", OS.GetUserDataDir());
    public static Variant GetSetting(string path, Variant defaultValue = default) => defaultValue;
    public static bool HasSetting(string path) => false;

    public static bool LoadResourcePack(string path, bool replaceFiles = true) => true;
}

// === Tween (no-op in headless) ===

public partial class Tween : RefCounted
{
    public enum TransitionType { Linear, Sine, Quint, Quart, Quad, Expo, Elastic, Cubic, Circ, Bounce, Back, Spring }
    public enum EaseType { In, Out, InOut, OutIn }

    public Tween SetTrans(TransitionType type) => this;
    public Tween SetEase(EaseType type) => this;
    public Tween SetParallel(bool parallel = true) => this;
    public Tween SetLoops(int loops = 0) => this;
    public PropertyTweener TweenProperty(GodotObject obj, NodePath property, Variant finalVal, double duration) => new();
    public CallbackTweener TweenCallback(Callable callback) => new();
    public IntervalTweener TweenInterval(double time) => new();
    public MethodTweener TweenMethod(Callable method, Variant from, Variant to, double duration) => new();
    public void Kill() { }
    public void FastForwardToCompletion() { }
    public bool IsRunning() => false;
    public bool IsValid() => false;
    public Tween Chain() => this;

    public event Action? Finished;

    public Tween Parallel() => this;
    public Tween Play() => this;
    public SignalAwaiter ToSignal(GodotObject source, StringName signal) => new(source, signal, this);
    public static class SignalName { public static readonly StringName Finished = "finished"; }
}

public class PropertyTweener : RefCounted
{
    public PropertyTweener SetTrans(Tween.TransitionType type) => this;
    public PropertyTweener SetEase(Tween.EaseType type) => this;
    public PropertyTweener SetDelay(double delay) => this;
    public PropertyTweener From(Variant value) => this;
    public PropertyTweener FromCurrent() => this;
    public PropertyTweener AsRelative() => this;
}

public class CallbackTweener : RefCounted
{
    public CallbackTweener SetDelay(double delay) => this;
}

public class IntervalTweener : RefCounted { }
public class MethodTweener : RefCounted
{
    public MethodTweener SetTrans(Tween.TransitionType type) => this;
    public MethodTweener SetEase(Tween.EaseType type) => this;
    public MethodTweener SetDelay(double delay) => this;
}

// === SceneTree (minimal, returned null from Engine.GetMainLoop) ===

public partial class SceneTree : GodotObject
{
    public Node Root => new();
    public SceneTreeTimer CreateTimer(double timeSec, bool processAlways = true, bool processInPhysics = false, bool ignoreTimeScale = false)
        => new();
    public void Quit(int exitCode = 0) { }
    public bool Paused { get; set; }

    public static class SignalName
    {
        public static readonly StringName ProcessFrame = "process_frame";
    }
}

public partial class SceneTreeTimer : RefCounted
{
    public double TimeLeft { get; set; }
    public event Action? Timeout;

    public static class SignalName
    {
        public static readonly StringName Timeout = "timeout";
    }
}

// === File I/O stubs ===

public partial class FileAccess : RefCounted, IDisposable
{
    public enum ModeFlags { Read = 1, Write = 2, ReadWrite = 3, WriteRead = 7 }
    private System.IO.FileStream? _stream;

    public static FileAccess? Open(string path, ModeFlags flags)
    {
        try
        {
            string fullPath = ProjectSettings.GlobalizePath(path);
            string? directory = System.IO.Path.GetDirectoryName(fullPath);
            if (!string.IsNullOrEmpty(directory) && !System.IO.Directory.Exists(directory))
            {
                System.IO.Directory.CreateDirectory(directory);
            }
            System.IO.FileMode mode = flags switch
            {
                ModeFlags.Read => System.IO.FileMode.Open,
                ModeFlags.Write => System.IO.FileMode.Create,
                ModeFlags.ReadWrite => System.IO.FileMode.OpenOrCreate,
                ModeFlags.WriteRead => System.IO.FileMode.Create,
                _ => System.IO.FileMode.OpenOrCreate,
            };
            System.IO.FileAccess access = flags switch
            {
                ModeFlags.Read => System.IO.FileAccess.Read,
                ModeFlags.Write => System.IO.FileAccess.Write,
                _ => System.IO.FileAccess.ReadWrite,
            };
            return new FileAccess
            {
                _stream = new System.IO.FileStream(fullPath, mode, access, System.IO.FileShare.ReadWrite),
            };
        }
        catch
        {
            return null;
        }
    }
    public static bool FileExists(string path) => System.IO.File.Exists(path);
    public static Error GetOpenError() => Error.Ok;
    public string GetAsText(bool skipCr = false)
    {
        if (_stream == null)
        {
            return "";
        }
        _stream.Position = 0;
        using System.IO.StreamReader reader = new System.IO.StreamReader(_stream, leaveOpen: true);
        string text = reader.ReadToEnd();
        return skipCr ? text.Replace("\r", "") : text;
    }
    public void StoreString(string value)
    {
        StoreBuffer(System.Text.Encoding.UTF8.GetBytes(value));
    }
    public byte[] GetBuffer(long length)
    {
        if (_stream == null)
        {
            return Array.Empty<byte>();
        }
        byte[] buffer = new byte[length];
        int read = _stream.Read(buffer, 0, (int)length);
        if (read == buffer.Length)
        {
            return buffer;
        }
        Array.Resize(ref buffer, read);
        return buffer;
    }
    public void StoreBuffer(byte[] buffer)
    {
        _stream?.Write(buffer, 0, buffer.Length);
    }
    public ulong GetLength() => (ulong)(_stream?.Length ?? 0);
    public ulong GetPosition() => (ulong)(_stream?.Position ?? 0);
    public bool IsOpen() => _stream != null;
    public void Seek(ulong position)
    {
        if (_stream != null)
        {
            _stream.Position = (long)position;
        }
    }
    public void Close()
    {
        _stream?.Dispose();
        _stream = null;
    }
    public void Flush() => _stream?.Flush();
    public static ulong GetModifiedTime(string path)
    {
        try
        {
            return (ulong)new DateTimeOffset(System.IO.File.GetLastWriteTimeUtc(ProjectSettings.GlobalizePath(path))).ToUnixTimeSeconds();
        }
        catch
        {
            return 0;
        }
    }
    public void Dispose() => Close();
}

public partial class DirAccess : RefCounted, IDisposable
{
    private string _path = "";
    private string[] _entries = Array.Empty<string>();
    private int _entryIndex;

    public bool IncludeHidden { get; set; }

    public static DirAccess? Open(string path)
    {
        string fullPath = ProjectSettings.GlobalizePath(path);
        if (!System.IO.Directory.Exists(fullPath))
        {
            return null;
        }
        return new DirAccess { _path = fullPath };
    }
    public string GetCurrentDir() => _path;
    public string[] GetFiles() => System.IO.Directory.Exists(_path) ? System.IO.Directory.GetFiles(_path).Select(System.IO.Path.GetFileName).Where(static f => f != null).Cast<string>().ToArray() : Array.Empty<string>();
    public string[] GetDirectories() => System.IO.Directory.Exists(_path) ? System.IO.Directory.GetDirectories(_path).Select(System.IO.Path.GetFileName).Where(static f => f != null).Cast<string>().ToArray() : Array.Empty<string>();
    public bool CurrentIsDir() => false;
    public bool DirExists(string path) => System.IO.Directory.Exists(ProjectSettings.GlobalizePath(path));
    public bool FileExists(string path) => System.IO.File.Exists(System.IO.Path.Combine(_path, path));
    public Error MakeDir(string path) => Error.Ok;
    public Error MakeDirRecursive(string path)
    {
        System.IO.Directory.CreateDirectory(ProjectSettings.GlobalizePath(path));
        return Error.Ok;
    }
    public Error Remove(string path)
    {
        try
        {
            string fullPath = string.IsNullOrEmpty(path) ? _path : System.IO.Path.Combine(_path, path);
            if (System.IO.File.Exists(fullPath))
            {
                System.IO.File.Delete(fullPath);
            }
            else if (System.IO.Directory.Exists(fullPath))
            {
                System.IO.Directory.Delete(fullPath, false);
            }
            return Error.Ok;
        }
        catch
        {
            return Error.Failed;
        }
    }
    public Error ListDirBegin(bool skipNavigational = false, bool skipHidden = false)
    {
        if (!System.IO.Directory.Exists(_path))
        {
            _entries = Array.Empty<string>();
            _entryIndex = 0;
            return Error.DoesNotExist;
        }
        _entries = System.IO.Directory.GetFileSystemEntries(_path).Select(System.IO.Path.GetFileName).Where(static f => f != null).Cast<string>().ToArray();
        _entryIndex = 0;
        return Error.Ok;
    }
    public string GetNext()
    {
        if (_entryIndex >= _entries.Length)
        {
            return "";
        }
        return _entries[_entryIndex++];
    }
    public void Dispose() { }
    public static bool DirExistsAbsolute(string path) => System.IO.Directory.Exists(ProjectSettings.GlobalizePath(path));
    public static Error RemoveAbsolute(string path)
    {
        try
        {
            string fullPath = ProjectSettings.GlobalizePath(path);
            if (System.IO.File.Exists(fullPath))
            {
                System.IO.File.Delete(fullPath);
            }
            else if (System.IO.Directory.Exists(fullPath))
            {
                System.IO.Directory.Delete(fullPath, true);
            }
            return Error.Ok;
        }
        catch
        {
            return Error.Failed;
        }
    }
    public static Error RenameAbsolute(string sourcePath, string destinationPath)
    {
        try
        {
            string source = ProjectSettings.GlobalizePath(sourcePath);
            string destination = ProjectSettings.GlobalizePath(destinationPath);
            string? directory = System.IO.Path.GetDirectoryName(destination);
            if (!string.IsNullOrEmpty(directory) && !System.IO.Directory.Exists(directory))
            {
                System.IO.Directory.CreateDirectory(directory);
            }
            System.IO.File.Move(source, destination, true);
            return Error.Ok;
        }
        catch
        {
            return Error.Failed;
        }
    }
    public static string[] GetFilesAt(string path)
    {
        string fullPath = ProjectSettings.GlobalizePath(path);
        return System.IO.Directory.Exists(fullPath)
            ? System.IO.Directory.GetFiles(fullPath).Select(System.IO.Path.GetFileName).Where(static f => f != null).Cast<string>().ToArray()
            : Array.Empty<string>();
    }
    public static string[] GetDirectoriesAt(string path)
    {
        string fullPath = ProjectSettings.GlobalizePath(path);
        return System.IO.Directory.Exists(fullPath)
            ? System.IO.Directory.GetDirectories(fullPath).Select(System.IO.Path.GetFileName).Where(static f => f != null).Cast<string>().ToArray()
            : Array.Empty<string>();
    }
    public static Error MakeDirRecursiveAbsolute(string path)
    {
        System.IO.Directory.CreateDirectory(ProjectSettings.GlobalizePath(path));
        return Error.Ok;
    }
    public static Error MakeDirAbsolute(string path)
    {
        System.IO.Directory.CreateDirectory(ProjectSettings.GlobalizePath(path));
        return Error.Ok;
    }
}

// === Resource loading stubs ===

public class PackedScene : Resource
{
    public Node? Instantiate(GenEditState editState = GenEditState.Disabled) => null;
    public T? Instantiate<T>(GenEditState editState = GenEditState.Disabled) where T : class => default;
    public enum GenEditState { Disabled, Instance, Main, MainInherited }
}

public static partial class ResourceLoader
{
    public static Resource? Load(string path, string typeHint = "", CacheMode cacheMode = CacheMode.Reuse) => null;
    public static T? Load<T>(string path, string typeHint = "", CacheMode cacheMode = CacheMode.Reuse) where T : class => null;
    public static bool Exists(string path, string typeHint = "") => false;
    public enum CacheMode { Reuse, Ignore, Replace }
    public enum ThreadLoadStatus { InvalidResource, InProgress, Failed, Loaded }

    public static ThreadLoadStatus LoadThreadedGetStatus(string? path) => ThreadLoadStatus.Loaded;
    public static Resource? LoadThreadedGet(string? path) => Load(path ?? "");
    public static Error LoadThreadedRequest(string path, string typeHint = "", bool useSubThreads = false, CacheMode cacheMode = CacheMode.Reuse) => Error.Ok;
    public static void AddResourceFormatLoader(ResourceFormatLoader loader, bool atFront = false) { }
}

public static class ResourceSaver
{
    public static Error Save(Resource resource, string path = "", SaverFlags flags = SaverFlags.None) => Error.Ok;
    public enum SaverFlags { None = 0 }
}

// === Error enum ===

public enum Error
{
    Ok = 0, Failed = 1, Unavailable = 2, Unconfigured = 3,
    Unauthorized = 4, ParameterRangeError = 5, OutOfMemory = 6,
    FileNotFound = 7, FileBadDrive = 8, FileBadPath = 9,
    FileNoPermission = 10, FileAlreadyInUse = 11, FileCantOpen = 12,
    FileCantWrite = 13, FileCantRead = 14, FileUnrecognized = 15,
    FileCorrupt = 16, FileMissingDependencies = 17, FileEof = 18,
    CantOpen = 19, CantCreate = 20, QueryFailed = 21, AlreadyInUse = 22,
    Locked = 23, Timeout = 24, CantConnect = 25, CantResolve = 26,
    ConnectionError = 27, CantAcquireResource = 28, CantFork = 29,
    InvalidData = 30, InvalidParameter = 31, AlreadyExists = 32,
    DoesNotExist = 33, DatabaseCantRead = 34, DatabaseCantWrite = 35,
    CompilationFailed = 36, MethodNotFound = 37, LinkFailed = 38,
    ScriptFailed = 39, CyclicLink = 40, InvalidDeclaration = 41,
    DuplicateSymbol = 42, ParseError = 43, Busy = 44,
    Skip = 45, Help = 46, Bug = 47, PrinterOnFire = 48,
}
