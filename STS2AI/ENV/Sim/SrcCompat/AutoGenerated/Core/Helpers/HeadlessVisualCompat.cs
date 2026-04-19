using Godot;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Models.Encounters;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;

namespace MegaCrit.Sts2.Core.Helpers;

public static class HeadlessVisualCompat
{
	public static NCreature? TryGetCreatureNode(Creature creature)
	{
		return NCombatRoom.Instance?.GetCreatureNode(creature);
	}

	public static T? TryGetSpecialNode<T>(NCreature? creatureNode, string path) where T : Node
	{
		return creatureNode?.GetSpecialNode<T>(path);
	}

	public static T? TryGetCurrentBodyNode<T>(NCreature? creatureNode, string path) where T : Node
	{
		return (creatureNode?.Visuals?.GetCurrentBody() as Node)?.GetNodeOrNull<T>(path);
	}

	public static T? TryGetBackgroundNode<T>(string path) where T : Node
	{
		return NCombatRoom.Instance?.Background?.GetNodeOrNull<T>(path);
	}

	public static Vector2 GetGlobalPositionOrZero(Node2D? node)
	{
		return node?.GlobalPosition ?? Vector2.Zero;
	}

	public static Vector2 GetGlobalPositionOrZero(NCreature? creatureNode)
	{
		return creatureNode?.GlobalPosition ?? Vector2.Zero;
	}

	public static Vector2 GetBodyScaleOrOne(NCreature? creatureNode)
	{
		return creatureNode?.Body?.Scale ?? Vector2.One;
	}

	public static Vector2 GetCurrentBodyScaleOrOne(NCreature? creatureNode)
	{
		return (creatureNode?.Visuals?.GetCurrentBody() as Node2D)?.Scale ?? Vector2.One;
	}

	public static float GetCurrentBodyScaleXOrOne(NCreature? creatureNode)
	{
		return (creatureNode?.Visuals?.GetCurrentBody() as Node2D)?.Scale.X ?? 1f;
	}

	public static void TrySetPosition(Node2D? node, Vector2 value)
	{
		if (node != null)
		{
			node.Position = value;
		}
	}

	public static void TryOffsetPosition(Node2D? node, Vector2 delta)
	{
		if (node != null)
		{
			node.Position += delta;
		}
	}

	public static void TrySetGlobalPosition(Node2D? node, Vector2 value)
	{
		if (node != null)
		{
			node.GlobalPosition = value;
		}
	}

	public static void TrySetGlobalPosition(NCreature? creatureNode, Vector2 value)
	{
		if (creatureNode != null)
		{
			creatureNode.GlobalPosition = value;
		}
	}

	public static void TryAddChild(Node? parent, Node? child)
	{
		if (parent != null && child != null)
		{
			parent.AddChildSafely(child);
		}
	}

	public static void TryAddCombatVfx(Node node)
	{
		NCombatRoom.Instance?.CombatVfxContainer?.AddChildSafely(node);
	}

	public static void TryAddBackCombatVfx(Node node)
	{
		NCombatRoom.Instance?.BackCombatVfxContainer?.AddChildSafely(node);
	}

	public static void TryAddGlobalUi(Node node)
	{
		NGame.Instance?.CurrentRunNode?.GlobalUi?.AddChildSafely(node);
	}

	public static void TrySetFabricatorBotFallPosition(NCreature? creatureNode)
	{
		if (creatureNode != null)
		{
			FabricatorNormal.SetBotFallPosition(creatureNode);
		}
	}

	public static void TryScreenShake(ShakeStrength strength, ShakeDuration duration)
	{
		NGame.Instance?.ScreenShake(strength, duration);
	}

	public static void TryScreenShake(ShakeStrength strength, ShakeDuration duration, float angle)
	{
		NGame.Instance?.ScreenShake(strength, duration, angle);
	}

	public static void TryScreenShakeTrauma(ShakeStrength strength)
	{
		NGame.Instance?.ScreenShakeTrauma(strength);
	}

	public static void TryScreenRumble(ShakeStrength strength, ShakeDuration duration, RumbleStyle style)
	{
		NGame.Instance?.ScreenRumble(strength, duration, style);
	}

	public static void TryDoHitStop(ShakeStrength strength, ShakeDuration duration)
	{
		NGame.Instance?.DoHitStop(strength, duration);
	}

	public static Vector2 GetViewportCenterOrZero()
	{
		return NGame.Instance?.GetViewportRect().Size / 2f ?? Vector2.Zero;
	}
}
