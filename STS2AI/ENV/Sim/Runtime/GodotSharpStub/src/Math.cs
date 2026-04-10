using System;

namespace Godot;

public struct Vector2
{
    public float X, Y;
    public Vector2(float x, float y) { X = x; Y = y; }
    public static readonly Vector2 Zero = new(0, 0);
    public static readonly Vector2 One = new(1, 1);
    public static readonly Vector2 Up = new(0, -1);
    public static readonly Vector2 Down = new(0, 1);
    public static readonly Vector2 Left = new(-1, 0);
    public static readonly Vector2 Right = new(1, 0);
    public float Length() => MathF.Sqrt(X * X + Y * Y);
    public float LengthSquared() => X * X + Y * Y;
    public Vector2 Normalized() { var l = Length(); return l > 0 ? new(X / l, Y / l) : Zero; }
    public float DistanceTo(Vector2 to) => (this - to).Length();
    public Vector2 Lerp(Vector2 to, float weight) => new(X + (to.X - X) * weight, Y + (to.Y - Y) * weight);
    public static Vector2 operator +(Vector2 a, Vector2 b) => new(a.X + b.X, a.Y + b.Y);
    public static Vector2 operator -(Vector2 a, Vector2 b) => new(a.X - b.X, a.Y - b.Y);
    public static Vector2 operator *(Vector2 a, float s) => new(a.X * s, a.Y * s);
    public static Vector2 operator *(float s, Vector2 a) => new(a.X * s, a.Y * s);
    public static Vector2 operator *(Vector2 a, Vector2 b) => new(a.X * b.X, a.Y * b.Y);
    public static Vector2 operator /(Vector2 a, float s) => new(a.X / s, a.Y / s);
    public static Vector2 operator /(Vector2 a, Vector2 b) => new(a.X / b.X, a.Y / b.Y);
    public static Vector2 operator -(Vector2 a) => new(-a.X, -a.Y);
    public static bool operator ==(Vector2 a, Vector2 b) => a.X == b.X && a.Y == b.Y;
    public static bool operator !=(Vector2 a, Vector2 b) => !(a == b);
    public override bool Equals(object? obj) => obj is Vector2 v && this == v;
    public override int GetHashCode() => HashCode.Combine(X, Y);
    public override string ToString() => $"({X}, {Y})";
}

public struct Vector2I
{
    public int X, Y;
    public Vector2I(int x, int y) { X = x; Y = y; }
    public static readonly Vector2I Zero = new(0, 0);
    public static readonly Vector2I One = new(1, 1);
    public static Vector2I operator +(Vector2I a, Vector2I b) => new(a.X + b.X, a.Y + b.Y);
    public static Vector2I operator -(Vector2I a, Vector2I b) => new(a.X - b.X, a.Y - b.Y);
    public static bool operator ==(Vector2I a, Vector2I b) => a.X == b.X && a.Y == b.Y;
    public static bool operator !=(Vector2I a, Vector2I b) => !(a == b);
    public static implicit operator Vector2(Vector2I v) => new(v.X, v.Y);
    public override bool Equals(object? obj) => obj is Vector2I v && this == v;
    public override int GetHashCode() => HashCode.Combine(X, Y);
}

public struct Vector3
{
    public float X, Y, Z;
    public Vector3(float x, float y, float z) { X = x; Y = y; Z = z; }
    public static readonly Vector3 Zero = new(0, 0, 0);
    public static readonly Vector3 One = new(1, 1, 1);
    public static Vector3 operator +(Vector3 a, Vector3 b) => new(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
    public static Vector3 operator -(Vector3 a, Vector3 b) => new(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
    public static Vector3 operator *(Vector3 a, float s) => new(a.X * s, a.Y * s, a.Z * s);
}

public struct Vector4
{
    public float X, Y, Z, W;
    public Vector4(float x, float y, float z, float w) { X = x; Y = y; Z = z; W = w; }
}

public struct Rect2
{
    public Vector2 Position, Size;
    public Rect2(Vector2 position, Vector2 size) { Position = position; Size = size; }
    public Rect2(float x, float y, float w, float h) { Position = new(x, y); Size = new(w, h); }
    public Vector2 End => Position + Size;
    public bool HasPoint(Vector2 point) =>
        point.X >= Position.X && point.Y >= Position.Y &&
        point.X < Position.X + Size.X && point.Y < Position.Y + Size.Y;
    public Rect2 Grow(float amount) => new(Position.X - amount, Position.Y - amount, Size.X + amount * 2, Size.Y + amount * 2);
}

public struct Rect2I
{
    public Vector2I Position, Size;
    public Rect2I(int x, int y, int w, int h) { Position = new(x, y); Size = new(w, h); }
}

public struct Transform2D
{
    public Vector2 X, Y, Origin;
    public static readonly Transform2D Identity = new() { X = new(1, 0), Y = new(0, 1) };
    public Transform2D(float rotation, Vector2 origin) { X = new(MathF.Cos(rotation), MathF.Sin(rotation)); Y = new(-MathF.Sin(rotation), MathF.Cos(rotation)); Origin = origin; }
    public static Vector2 operator *(Transform2D transform, Vector2 vector) => new(
        transform.X.X * vector.X + transform.Y.X * vector.Y + transform.Origin.X,
        transform.X.Y * vector.X + transform.Y.Y * vector.Y + transform.Origin.Y);
}

public struct Color
{
    public float R, G, B, A;
    public Color(float r, float g, float b, float a = 1f) { R = r; G = g; B = b; A = a; }
    public Color(Color color) { R = color.R; G = color.G; B = color.B; A = color.A; }
    public Color(string htmlColor) { R = 1; G = 1; B = 1; A = 1; } // Simplified
    public static Color Color8(byte r, byte g, byte b, byte a = 255) => new(r / 255f, g / 255f, b / 255f, a / 255f);
    public Color Lerp(Color to, float weight) => new(R + (to.R - R) * weight, G + (to.G - G) * weight, B + (to.B - B) * weight, A + (to.A - A) * weight);
    public static readonly Color White = new(1, 1, 1, 1);
    public static readonly Color Black = new(0, 0, 0, 1);
    public static readonly Color Transparent = new(0, 0, 0, 0);
    public static bool operator ==(Color a, Color b) => a.R == b.R && a.G == b.G && a.B == b.B && a.A == b.A;
    public static bool operator !=(Color a, Color b) => !(a == b);
    public override bool Equals(object? obj) => obj is Color c && this == c;
    public override int GetHashCode() => HashCode.Combine(R, G, B, A);
}

public static partial class Colors
{
    public static readonly Color White = Color.White;
    public static readonly Color Black = Color.Black;
    public static readonly Color Red = new(1, 0, 0);
    public static readonly Color Green = new(0, 1, 0);
    public static readonly Color Blue = new(0, 0, 1);
    public static readonly Color Yellow = new(1, 1, 0);
    public static readonly Color Transparent = Color.Transparent;
    public static readonly Color Magenta = new(1, 0, 1);
    public static readonly Color DarkRed = new(0.55f, 0, 0);
    public static readonly Color Purple = new(0.5f, 0, 0.5f);
}

public static partial class Mathf
{
    public const float Pi = MathF.PI;
    public const float Tau = MathF.PI * 2;
    public const float Inf = float.PositiveInfinity;
    public const float Epsilon = 1e-6f;
    public static float Abs(float x) => MathF.Abs(x);
    public static int Abs(int x) => Math.Abs(x);
    public static float Sin(float x) => MathF.Sin(x);
    public static float Cos(float x) => MathF.Cos(x);
    public static float Sqrt(float x) => MathF.Sqrt(x);
    public static float Floor(float x) => MathF.Floor(x);
    public static float Ceil(float x) => MathF.Ceiling(x);
    public static float Round(float x) => MathF.Round(x);
    public static int RoundToInt(float x) => (int)MathF.Round(x);
    public static int RoundToInt(double x) => (int)Math.Round(x);
    public static int FloorToInt(float x) => (int)MathF.Floor(x);
    public static int FloorToInt(double x) => (int)Math.Floor(x);
    public static int CeilToInt(float x) => (int)MathF.Ceiling(x);
    public static int CeilToInt(double x) => (int)Math.Ceiling(x);
    public static float Clamp(float value, float min, float max) => Math.Clamp(value, min, max);
    public static int Clamp(int value, int min, int max) => Math.Clamp(value, min, max);
    public static float Lerp(float from, float to, float weight) => from + (to - from) * weight;
    public static float InverseLerp(float from, float to, float value) => (value - from) / (to - from);
    public static float Min(float a, float b) => MathF.Min(a, b);
    public static float Max(float a, float b) => MathF.Max(a, b);
    public static int Min(int a, int b) => Math.Min(a, b);
    public static int Max(int a, int b) => Math.Max(a, b);
    public static float Pow(float b, float e) => MathF.Pow(b, e);
    public static float Log(float x) => MathF.Log(x);
    public static float Sign(float x) => MathF.Sign(x);
    public static bool IsEqualApprox(float a, float b) => MathF.Abs(a - b) < Epsilon;
    public static bool IsZeroApprox(float x) => MathF.Abs(x) < Epsilon;
    public static float MoveToward(float from, float to, float delta) => MathF.Abs(to - from) <= delta ? to : from + MathF.Sign(to - from) * delta;
    public static float Wrap(float value, float min, float max) { var r = max - min; return min + ((value - min) % r + r) % r; }
    public static int Wrap(int value, int min, int max) { var r = max - min; return min + ((value - min) % r + r) % r; }
    public static float SmoothStep(float from, float to, float x) { var t = Clamp((x - from) / (to - from), 0, 1); return t * t * (3 - 2 * t); }
    public static float Snapped(float value, float step) => step != 0 ? MathF.Round(value / step) * step : value;
    public static double Snapped(double value, double step) => step != 0 ? Math.Round(value / step) * step : value;
    public static float DegToRad(float deg) => deg * Pi / 180f;
    public static float RadToDeg(float rad) => rad * 180f / Pi;
    public static float Atan2(float y, float x) => MathF.Atan2(y, x);
    public static float Atan2(double y, double x) => MathF.Atan2((float)y, (float)x);
}

public struct Quaternion
{
    public float X, Y, Z, W;
    public static readonly Quaternion Identity = new() { W = 1 };
}

public struct Basis
{
    public Vector3 X, Y, Z;
    public static readonly Basis Identity = new() { X = new(1, 0, 0), Y = new(0, 1, 0), Z = new(0, 0, 1) };
}

public struct Transform3D
{
    public Basis Basis;
    public Vector3 Origin;
    public static readonly Transform3D Identity = new() { Basis = Basis.Identity };
}
