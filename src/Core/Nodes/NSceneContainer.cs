using Godot;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.Nodes;

public partial class NSceneContainer : Control
{
	private Control? _currentScene;

	public Control? CurrentScene
	{
		get
		{
			if (_currentScene == null)
			{
				return null;
			}
			if (!GodotObject.IsInstanceValid(_currentScene))
			{
				return null;
			}
			if (_currentScene.IsQueuedForDeletion())
			{
				return null;
			}
			return _currentScene;
		}
		private set
		{
			_currentScene = value;
		}
	}

	public void SetCurrentScene(Control node)
	{
		foreach (Node child in GetChildren())
		{
			this.RemoveChildSafely(child);
			child.QueueFreeSafely();
		}
		CurrentScene = node;
		if (node.GetParent() == null)
		{
			this.AddChildSafely(node);
		}
		else
		{
			node.Reparent(this);
		}
	}
}
