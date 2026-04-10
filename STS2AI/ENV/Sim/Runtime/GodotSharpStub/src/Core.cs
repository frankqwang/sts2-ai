// GodotSharp stub — minimal types for headless simulation.
// Only provides enough to compile src/Core/ without the real Godot engine.

using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Godot;

// === Base Types ===

public class GodotObject
{
    public virtual void _Ready() { }
    public virtual void _Process(double delta) { }
    public virtual void _EnterTree() { }
    public virtual void _ExitTree() { }
    public virtual void _Notification(int what) { }
    public virtual bool _Set(StringName property, Variant value) => false;
    public virtual Variant _Get(StringName property) => default;
    public bool IsInstanceValid() => true;
    public static bool IsInstanceValid(GodotObject? obj) => obj != null;
    public bool IsValid() => true;
    public void QueueFree() { }
    public void SetMeta(StringName name, Variant value) { }
    public Variant GetMeta(StringName name, Variant def = default) => def;
    public bool HasMeta(StringName name) => false;
    public void SetProcess(bool enable) { }
    public void SetProcessInput(bool enable) { }
    public ulong GetInstanceId() => 0;
    public void Free() { }
    public SignalAwaiter ToSignal(GodotObject? source, StringName signal) => new(source, signal, this);
    public void EmitSignal(StringName signal, params object?[] args) { }
    public Error Connect(StringName signal, Callable callable) => Error.Ok;
    public void Disconnect(StringName signal, Callable callable) { }
}

public partial class Node : GodotObject
{
    public string Name { get; set; } = "";
    public Node? GetParent() => null;
    public Node? GetNode(NodePath path) => null;
    public Node? GetNodeOrNull(NodePath path) => null;
    public T? GetNode<T>(NodePath path) where T : class => null;
    public T? GetNodeOrNull<T>(NodePath path) where T : class => null;
    public Node? FindChild(string name, bool recursive = true, bool owned = true) => null;
    public T? FindChild<T>(string name, bool recursive = true, bool owned = true) where T : class => null;
    public Godot.Collections.Array<Node> GetChildren(bool includeInternal = false) => new();
    public int GetChildCount(bool includeInternal = false) => 0;
    public Node? GetChild(int idx, bool includeInternal = false) => null;
    public T? GetChild<T>(int idx) where T : class => null;
    public void AddChild(Node child, bool forceReadableName = false, InternalMode internalMode = InternalMode.Disabled) { }
    public void RemoveChild(Node child) { }
    public void MoveChild(Node child, int toIndex) { }
    public int GetIndex(bool includeInternal = false) => 0;
    public bool IsInsideTree() => false;
    public SceneTree? GetTree() => null;
    public Viewport GetViewport() => new();
    public Rect2 GetViewportRect() => GetViewport().GetVisibleRect();
    public Tween? CreateTween() => null;
    public void SetPhysicsProcess(bool enable) { }
    public bool IsNodeReady() => true;
    public NodePath GetPath() => new NodePath("");
    public void Reparent(Node newParent, bool keepGlobalTransform = true) { }
    public bool HasNode(NodePath path) => false;
    public bool IsAncestorOf(Node? node) => false;
    public virtual string _GetConfigurationWarnings() => "";

    public enum InternalMode { Disabled, Front, Back }

    // Signal-like event stubs
    public event Action? TreeEntered;
    public event Action? TreeExited;
    public event Action? Ready;
    public void CallDeferred(object? a = null, object? b = null, object? c = null, object? d = null) { }
    public double GetProcessDeltaTime() => 0;

    public static class SignalName
    {
        public static readonly StringName TreeEntered = "tree_entered";
        public static readonly StringName TreeExited = "tree_exited";
        public static readonly StringName TreeExiting = "tree_exiting";
        public static readonly StringName ChildEnteredTree = "child_entered_tree";
        public static readonly StringName ChildExitingTree = "child_exiting_tree";
        public static readonly StringName Ready = "ready";
        public static readonly StringName Pressed = "pressed";
        public static readonly StringName NodeHovered = "node_hovered";
        public static readonly StringName NodeUnhovered = "node_unhovered";
        public static readonly StringName HitCreature = "hit_creature";
        public static readonly StringName Changed = "changed";
        public static readonly StringName VisibilityChanged = "visibility_changed";
        public static readonly StringName Completed = "completed";
    }

    public static class MethodName
    {
        public static readonly StringName QueueFree = "queue_free";
        public static readonly StringName Show = "show";
        public static readonly StringName Hide = "hide";
        public static readonly StringName AddChild = "add_child";
        public static readonly StringName RemoveChild = "remove_child";
    }
}

public partial class Node2D : Node
{
    public Vector2 Position { get; set; }
    public Vector2 GlobalPosition { get; set; }
    public float Rotation { get; set; }
    public Vector2 Scale { get; set; } = Vector2.One;
    public bool Visible { get; set; } = true;
    public int ZIndex { get; set; }
    public void Show() { Visible = true; }
    public void Hide() { Visible = false; }
    public void SetVisible(bool visible) { Visible = visible; }
    public float RotationDegrees { get; set; }
}

public partial class Control : Node
{
    public enum FocusModeEnum { None, Click, All }
    public Vector2 Position { get; set; }
    public Vector2 GlobalPosition { get; set; }
    public Vector2 Size { get; set; }
    public Vector2 CustomMinimumSize { get; set; }
    public bool Visible { get; set; } = true;
    public Color Modulate { get; set; } = Colors.White;
    public MouseFilterEnum MouseFilter { get; set; } = MouseFilterEnum.Stop;
    public NodePath FocusNeighborTop { get; set; }
    public NodePath FocusNeighborBottom { get; set; }
    public NodePath FocusNeighborLeft { get; set; }
    public NodePath FocusNeighborRight { get; set; }
    public FocusModeEnum FocusMode { get; set; } = FocusModeEnum.None;
    public void Show() { Visible = true; }
    public void Hide() { Visible = false; }
    public void GrabFocus() { }
    public void SetFocusMode(FocusModeEnum mode) => FocusMode = mode;
    public bool TryGrabFocus() { GrabFocus(); return true; }
    public bool HasFocus() => false;
    public Vector2 GetGlobalMousePosition() => Vector2.Zero;

    public enum LayoutPreset { TopLeft, TopRight, BottomLeft, BottomRight, FullRect }
    public enum MouseFilterEnum { Stop, Pass, Ignore }
    public void SetAnchorsAndOffsetsPreset(LayoutPreset preset) { }
    public void AddThemeFontOverride(object? a = null, object? b = null, object? c = null, object? d = null) { }
    public Transform2D GetGlobalTransformWithCanvas() => Transform2D.Identity;
}

public partial class CanvasLayer : Node
{
    public int Layer { get; set; }
}

public partial class CanvasItem : Node2D
{
    public Color Modulate { get; set; } = Colors.White;
    public Color SelfModulate { get; set; } = Colors.White;
    public bool ClipChildren { get; set; }
}

public partial class Resource : GodotObject
{
    public string ResourcePath { get; set; } = "";
    public string ResourceName { get; set; } = "";
    public Resource Duplicate(bool subresources = false, bool deep = false) => this;

        public void Dispose(object? a = null, object? b = null, object? c = null, object? d = null) { }
    }

public class ResourceFormatLoader : RefCounted
{
    public virtual string[] _GetRecognizedExtensions() => Array.Empty<string>();
    public virtual bool _HandlesType(StringName type) => false;
    public virtual string _GetResourceType(string path) => "";
    public virtual bool _RecognizePath(string path, StringName type) => false;
    public virtual bool _Exists(string path) => false;
    public virtual Variant _Load(string path, string originalPath, bool useSubThreads, int cacheMode) => default;
    public virtual string[] _GetDependencies(string path, bool addTypes) => Array.Empty<string>();
}

public class RefCounted : GodotObject { }

// === Variant ===

public struct Variant
{
    private object? _value;
    public static implicit operator Variant(int v) => new() { _value = v };
    public static implicit operator Variant(float v) => new() { _value = v };
    public static implicit operator Variant(double v) => new() { _value = v };
    public static implicit operator Variant(string v) => new() { _value = v };
    public static implicit operator Variant(bool v) => new() { _value = v };
    public static implicit operator Variant(long v) => new() { _value = v };
    public static implicit operator Variant(Vector2 v) => new() { _value = v };
    public static implicit operator Variant(Vector2I v) => new() { _value = v };
    public static implicit operator Variant(Color v) => new() { _value = v };
    public static implicit operator Variant(Node? v) => new() { _value = v };
    public static implicit operator Variant(Resource? v) => new() { _value = v };
    public static implicit operator Variant(Texture2D? v) => new() { _value = v };
    public static implicit operator Variant(AtlasTexture? v) => new() { _value = v };
    public static implicit operator int(Variant v) => v._value is int i ? i : 0;
    public static implicit operator float(Variant v) => v._value is float f ? f : 0f;
    public static implicit operator string(Variant v) => v._value?.ToString() ?? "";
    public static implicit operator bool(Variant v) => v._value is bool b && b;
    public VariantType VariantType => VariantType.Nil;
    public T As<T>() => _value is T t ? t : default!;
    public object? Obj => _value;
}

public enum VariantType { Nil, Bool, Int, Float, String, Object }

// === StringName ===

public struct StringName
{
    private readonly string _name;
    public StringName(string name) { _name = name ?? ""; }
    public static implicit operator StringName(string s) => new(s);
    public static implicit operator string(StringName sn) => sn._name ?? "";
    public override string ToString() => _name ?? "";
    public override int GetHashCode() => (_name ?? "").GetHashCode();
    public override bool Equals(object? obj) => obj is StringName sn && _name == sn._name;
    public static bool operator ==(StringName a, StringName b) => a._name == b._name;
    public static bool operator !=(StringName a, StringName b) => a._name != b._name;
}

// === NodePath ===

public struct NodePath
{
    private readonly string _path;
    public NodePath(string path) { _path = path ?? ""; }
    public static implicit operator NodePath(string s) => new(s);
    public static implicit operator string(NodePath np) => np._path ?? "";
    public override string ToString() => _path ?? "";
    public bool IsEmpty => string.IsNullOrEmpty(_path);
}

// === Signal ===

[AttributeUsage(AttributeTargets.Event | AttributeTargets.Delegate)]
public class SignalAttribute : Attribute { }

public struct Signal
{
    public static Signal operator +(Signal s, Callable c) => s;
    public static Signal operator -(Signal s, Callable c) => s;
}

public partial struct Callable
{
    public static Callable From(Action a) => default;
    public static Callable From<T>(Action<T> a) => default;
    public static Callable From(Delegate d) => default;
    public void Call(params object[] args) { }
    public void CallDeferred(object? a = null, object? b = null, object? c = null, object? d = null) { }
}

public partial class Viewport : Node
{
    public Control? GuiGetFocusOwner() => null;
    public void GuiReleaseFocus() { }
    public void SetInputAsHandled() { }
    public Vector2 GetMousePosition() => Vector2.Zero;
    public Rect2 GetVisibleRect() => new(Vector2.Zero, new Vector2(1920, 1080));
    public ViewportTexture GetTexture() => new();
    public ulong GetViewportRid() => 0;

    public static class SignalName
    {
        public static readonly StringName SizeChanged = "size_changed";
        public static readonly StringName GuiFocusChanged = "gui_focus_changed";
    }
}

public static class StringExtensions
{
    public static string GetBaseDir(this string path)
    {
        if (string.IsNullOrEmpty(path))
            return "";
        path = path.Replace('\\', '/');
        var index = path.LastIndexOf('/');
        return index <= 0 ? "" : path[..index];
    }

    public static string ToSnakeCase(this string value)
    {
        if (string.IsNullOrEmpty(value))
            return value;
        var chars = new List<char>(value.Length + 8);
        for (var i = 0; i < value.Length; i++)
        {
            var ch = value[i];
            if (char.IsUpper(ch) && i > 0)
                chars.Add('_');
            chars.Add(char.ToLowerInvariant(ch));
        }
        return new string(chars.ToArray());
    }

    public static string Capitalize(this string value)
    {
        if (string.IsNullOrEmpty(value))
            return value;
        return char.ToUpperInvariant(value[0]) + value[1..];
    }
}

// === Export ===

[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
public class ExportAttribute : Attribute
{
    public ExportAttribute() { }
    public ExportAttribute(PropertyHint hint, string hintString = "") { }
}

[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
public class ExportGroupAttribute : Attribute
{
    public ExportGroupAttribute(string name, string prefix = "") { }
}

[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
public class ExportSubgroupAttribute : Attribute
{
    public ExportSubgroupAttribute(string name, string prefix = "") { }
}

[AttributeUsage(AttributeTargets.Class)]
public class GlobalClassAttribute : Attribute { }

[AttributeUsage(AttributeTargets.Class)]
public class ToolAttribute : Attribute { }

public enum PropertyHint
{
    None, Range, Enum, Flags, File, Dir, GlobalFile, GlobalDir,
    ResourceType, MultilineText, PlaceholderText, Expression,
    Layers2DPhysics, Layers2DRender
}

public enum PropertyUsageFlags
{
    None = 0, Storage = 1, Editor = 2, Default = 7
}
