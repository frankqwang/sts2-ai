// Stub Godot UI/rendering types that game logic references but never uses in headless mode.

using System;
using System.Runtime.CompilerServices;

namespace Godot;

// === Visual types ===
public class Texture2D : Resource { public virtual Image GetImage() => new(); }
public class CompressedTexture2D : Texture2D { }
public partial class AtlasTexture : Texture2D
{
    public Rect2 Region { get; set; }
    public Rect2 Margin { get; set; }
    public Texture2D? Atlas { get; set; }
}
public class ImageTexture : Texture2D { }
public class GradientTexture2D : Texture2D { }
public class NoiseTexture2D : Texture2D { }
public class ViewportTexture : Texture2D { public override Image GetImage() => new(); }
public class Texture : Resource { }
public class Image : Resource
{
    public static Image? CreateEmpty(int w, int h, bool mips, Format format) => null;
    public enum Format { Rgba8, Rgb8 }
}
public class Material : Resource { }
public class ShaderMaterial : Material
{
    public void SetShaderParameter(StringName param, Variant value) { }
    public Variant GetShaderParameter(StringName param) => default;
}
public class Shader : Resource { }
public class StyleBox : Resource { }
public class StyleBoxFlat : StyleBox { }
public class Font : Resource { }
public class FontFile : Font { }
public class Theme : Resource { }
public class Curve : Resource { public float Sample(float t) => 0; }
public class Gradient : Resource { }

// === Node types used in Models ===
public partial class Sprite2D : Node2D { public Texture2D? Texture { get; set; } }
public partial class TextureRect : Control
{
    public enum ExpandModeEnum { KeepSize, IgnoreSize, FitWidth, FitWidthProportional, FitHeight, FitHeightProportional }

    public Texture2D? Texture { get; set; }
    public ExpandModeEnum ExpandMode { get; set; }
    public Material? Material { get; set; }
    public Color SelfModulate { get; set; } = Colors.White;

    public void SetAnchorsPreset(object? a = null, object? b = null, object? c = null, object? d = null) { }
}
public partial class ColorRect : Control { public Color Color { get; set; } }
public partial class Label : Control { public string Text { get; set; } = ""; }
public partial class RichTextLabel : Control { public string Text { get; set; } = ""; }
public partial class Button : Control { public string Text { get; set; } = ""; public event Action? Pressed; }
public partial class BaseButton : Control { public event Action? Pressed; }
public partial class TextureButton : BaseButton { }
public partial class LineEdit : Control { public string Text { get; set; } = ""; }
public partial class TextEdit : Control { public string Text { get; set; } = ""; }
public partial class Container : Control { }
public partial class BoxContainer : Container { }
public partial class VBoxContainer : BoxContainer { }
public partial class HBoxContainer : BoxContainer { }
public partial class MarginContainer : Container { }
public partial class PanelContainer : Container { }
public partial class CenterContainer : Container { }
public partial class GridContainer : Container { public int Columns { get; set; } }
public partial class ScrollContainer : Container { }
public partial class Panel : Control { }
public partial class SubViewport : Node { }
public partial class SubViewportContainer : Container { }
public partial class Window : Node { }
public partial class Popup : Window { }
public partial class PopupMenu : Popup { }
public partial class NinePatchRect : Control { }

// === 2D nodes ===
public partial class Line2D : Node2D { }
public partial class Polygon2D : Node2D { }
public partial class AnimatedSprite2D : Node2D { }
public partial class Area2D : Node2D { }
public partial class CollisionShape2D : Node2D { }
public partial class RayCast2D : Node2D { }
public partial class Camera2D : Node2D { }
public partial class RemoteTransform2D : Node2D { }
public partial class Path2D : Node2D { }
public partial class PathFollow2D : Node2D { }
public partial class Parallax2D : Node2D { }

// === Particles ===
public partial class GpuParticles2D : Node2D
{
    public bool Emitting { get; set; }
    public int Amount { get; set; }
    public double Lifetime { get; set; }
    public bool OneShot { get; set; }
    public Material? ProcessMaterial { get; set; }
}
public partial class CpuParticles2D : Node2D
{
    public bool Emitting { get; set; }
    public int Amount { get; set; }
    public double Lifetime { get; set; }
    public bool OneShot { get; set; }
}
public class ParticleProcessMaterial : Material { }

// === Audio ===
public partial class AudioStreamPlayer : Node
{
    public float VolumeDb { get; set; }
    public bool Playing { get; set; }
    public void Play(float fromPosition = 0) { }
    public void Stop() { }
}
public class AudioStream : Resource { }
public class AudioBusLayout : Resource { }

// === Input ===
public class InputEvent : Resource { public bool IsPressed() => false; }
public class InputEventAction : InputEvent { public StringName Action { get; set; } }
public class InputEventKey : InputEvent { public Key Keycode { get; set; } }
public class InputEventMouseButton : InputEvent { public MouseButton ButtonIndex { get; set; } }
public class InputEventJoypadButton : InputEvent { }
public class InputEventJoypadMotion : InputEvent { }

public static class Input
{
    public static bool IsActionPressed(StringName action) => false;
    public static bool IsActionJustPressed(StringName action) => false;
    public static Vector2 GetVector(StringName negX, StringName posX, StringName negY, StringName posY) => Vector2.Zero;
    public static void SetCustomMouseCursor(Resource? cursor, CursorShape shape = CursorShape.Arrow, Vector2 hotspot = default) { }
    public enum CursorShape { Arrow, Ibeam, PointingHand, Cross, Wait, Busy, Drag, CanDrop, Forbidden, Vsize, Hsize, Bdiagsize, Fdiagsize, Move, Vsplit, Hsplit, Help }
}

public enum Key { None, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, Escape, Enter, Space, Tab, Backspace, Delete, Shift, Ctrl, Alt }
public enum MouseButton { None, Left, Right, Middle, WheelUp, WheelDown }
public enum JoyAxis { LeftX, LeftY, RightX, RightY, TriggerLeft, TriggerRight }
public enum JoyButton { A, B, X, Y, Back, Guide, Start, LeftStick, RightStick, LeftShoulder, RightShoulder, DpadUp, DpadDown, DpadLeft, DpadRight }

// === Misc ===
public class Timer : Node
{
    public double WaitTime { get; set; }
    public bool OneShot { get; set; }
    public bool Autostart { get; set; }
    public void Start(double timeSec = -1) { }
    public void Stop() { }
    public event Action? Timeout;
}
public class AnimationPlayer : Node
{
    public void Play(StringName name = default, double customBlend = -1, float customSpeed = 1, bool fromEnd = false) { }
    public void Stop(bool keepState = false) { }
    public bool IsPlaying() => false;
    public event Action<StringName>? AnimationFinished;
}

public class WorldEnvironment : Node { }
// ResourceFormatLoader is in Core.cs (with virtual methods for AtlasResourceLoader overrides)
public class SignalAwaiter : INotifyCompletion
{
    public SignalAwaiter() { }
    public SignalAwaiter(GodotObject? source, StringName signal, GodotObject? owner = null) { }
    public SignalAwaiter GetAwaiter() => this;
    public bool IsCompleted => true;
    public void GetResult() { }
    public void OnCompleted(Action continuation) => continuation();
}

[AttributeUsage(AttributeTargets.Property | AttributeTargets.Field)]
public class ExportToolButtonAttribute : Attribute { public ExportToolButtonAttribute(string text, string icon = "") { } }

// === Rendering ===
public class RenderingServer
{
    public static void SetDefaultClearColor(Color color) { }
}

public static class TranslationServer
{
    public static string Translate(StringName message, StringName context = default) => message;
    public static void SetLocale(string locale) { }
}

public partial class DisplayServer
{
    public static Vector2I WindowGetSize(int id = 0) => new(1920, 1080);
    public static void WindowSetSize(Vector2I size, int id = 0) { }
    public static int GetScreenCount() => 1;
    public static string GetName() => "headless";
}

// Additional Godot types needed by generated stubs
public partial class BackBufferCopy : Node2D { }
public partial class InputEventPanGesture : InputEvent { public Vector2 Delta { get; set; } }
public partial class FlowContainer : Control { }
public partial class Marker2D : Node2D { }
public partial class CanvasGroup : Node2D
{
    public Color SelfModulate { get; set; } = Colors.White;
    public void SetSelfModulate(Color color) { SelfModulate = color; }
}

// Godot ENet networking types
public partial class ENetConnection : GodotObject
{
    public enum EventType { None = -1, Connect = 0, Disconnect = 1, Receive = 2, Error = 3 }
    public struct ServiceResult { public EventType type; public ENetPacketPeer? peer; public byte[]? packetData; public int error; }
    public bool TryService(out ServiceResult? output) { output = null; return false; }
    public void CreateHostBound(string address, int port, int maxPeers = 32, int maxChannels = 0, int inBandwidth = 0, int outBandwidth = 0) { }
    public void CreateHost(int maxPeers = 32, int maxChannels = 0, int inBandwidth = 0, int outBandwidth = 0) { }
    public ENetPacketPeer? ConnectToHost(string address, int port, int channels = 0, int data = 0) => null;
    public void Flush() { }
    public void Destroy() { }
    public Error Service(int timeout, ServiceResult output) => Error.Ok;
}
public partial class ENetPacketPeer : GodotObject
{
    public enum PeerState { Disconnected, Connecting, AcknowledgingConnect, ConnectionPending, ConnectionSucceeded, Connected, DisconnectLater, Disconnecting, AcknowledgingDisconnect, Zombie }
    public void SetTimeout(int limit, int min, int max) { }
    public void Reset() { }
    public Error Send(int channel, byte[] data, int flags) => Error.Ok;
    public void PeerDisconnect(int data = 0) { }
    public void PeerDisconnectNow(int data = 0) { }
    public void PeerDisconnectLater(int data = 0) { }
    public byte[] GetPacket() => System.Array.Empty<byte>();
    public Error GetPacketError() => Error.Ok;
    public PeerState GetState() => PeerState.Disconnected;
    public bool IsActive() => false;
}
public partial class ENetMultiplayerPeer : GodotObject { }
