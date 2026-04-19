using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Nodes.CommonUi;

namespace MegaCrit.Sts2.Core.Nodes.GodotExtensions;

public static class NodeUtil
{
	public static async Task<float> AwaitProcessFrame(this Node node, CancellationToken ct = default(CancellationToken))
	{
		ct.ThrowIfCancellationRequested();
		SceneTree tree = node.GetTree();
		if (tree == null)
		{
			throw new TaskCanceledException();
		}
		await node.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
		ct.ThrowIfCancellationRequested();
		if (!node.IsValid() || !node.IsInsideTree())
		{
			throw new TaskCanceledException();
		}
		return (float)node.GetProcessDeltaTime();
	}

	public static bool IsDescendant(Node parent, Node candidate)
	{
		for (Node parent2 = candidate.GetParent(); parent2 != null; parent2 = parent2.GetParent())
		{
			if (parent2 == parent)
			{
				return true;
			}
		}
		return false;
	}

	public static bool IsValid(this Node? node)
	{
		if (node != null && GodotObject.IsInstanceValid(node))
		{
			return !node.IsQueuedForDeletion();
		}
		return false;
	}

	public static void TryGrabFocus(this Control control)
	{
		if (NControllerManager.Instance.IsUsingController)
		{
			if (control.IsVisibleInTree())
			{
				control.GrabFocus();
			}
			else
			{
				Callable.From(control.GrabFocus).CallDeferred();
			}
		}
	}

	public static T? GetAncestorOfType<T>(this Node node)
	{
		for (Node parent = node.GetParent(); parent != null; parent = parent.GetParent())
		{
			if (parent is T)
			{
				return (T)(object)((parent is T) ? parent : null);
			}
		}
		return default(T);
	}

	public static Task AwaitSignal(this GodotObject source, StringName signal, Node owner)
	{
		if (!GodotObject.IsInstanceValid(source))
		{
			return Task.CompletedTask;
		}
		TaskCompletionSource tcs = new TaskCompletionSource();
		bool resolved = false;
		Callable callable = default(Callable);
		callable = Callable.From(OnSignal);
		source.Connect(signal, callable);
		owner.TreeExiting += OnExiting;
		return tcs.Task;
		void OnExiting()
		{
			if (!resolved)
			{
				resolved = true;
				if (GodotObject.IsInstanceValid(source))
				{
					source.Disconnect(signal, callable);
				}
				tcs.TrySetCanceled();
			}
		}
		void OnSignal()
		{
			if (!resolved)
			{
				resolved = true;
				if (GodotObject.IsInstanceValid(source))
				{
					source.Disconnect(signal, callable);
				}
				if (GodotObject.IsInstanceValid(owner))
				{
					owner.TreeExiting -= OnExiting;
				}
				tcs.TrySetResult();
			}
		}
	}

	public static IEnumerable<T> GetChildrenRecursive<T>(this Node node)
	{
		foreach (Node child in node.GetChildren())
		{
			foreach (T item in child.GetChildrenRecursive<T>())
			{
				yield return item;
			}
			if (child is T)
			{
				yield return (T)(object)((child is T) ? child : null);
			}
		}
	}
}
