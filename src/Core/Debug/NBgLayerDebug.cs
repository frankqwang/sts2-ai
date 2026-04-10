using System.Collections.Generic;
using System.IO;
using Godot;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.Debug;

[Tool]
public partial class NBgLayerDebug : Control
{
	public enum LayerVisibility
	{
		A,
		B,
		C
	}

	private const string _layerNodePrefix = "Layer_";

	private PackedScene? _layerA;

	private PackedScene? _layerB;

	private PackedScene? _layerC;

	private LayerVisibility _visibleLayer;

	[Export(PropertyHint.None, "")]
	public LayerVisibility VisibleLayer
	{
		get
		{
			return _visibleLayer;
		}
		set
		{
			_visibleLayer = value;
			if (Engine.IsEditorHint())
			{
				UpdateLayers();
			}
		}
	}

	[ExportToolButton("Reload Layers")]
	private Callable ReloadLayersCallable => Callable.From(ReloadLayers);

	public override void _EnterTree()
	{
		if (Engine.IsEditorHint())
		{
			ReloadLayers();
		}
	}

	private void ReloadLayers()
	{
		string sceneFilePath = GetTree().GetEditedSceneRoot().SceneFilePath;
		if (sceneFilePath == null)
		{
			return;
		}
		string text;
		if (base.Name.ToString() == "Foreground")
		{
			text = "fg";
		}
		else
		{
			string text2 = base.Name.ToString();
			int length = "Layer_".Length;
			if (!int.TryParse(text2.Substring(length, text2.Length - length), out var result))
			{
				return;
			}
			text = $"bg_{result:D2}";
		}
		string file = sceneFilePath.GetFile();
		string text3 = file.Substring(0, file.LastIndexOf('_'));
		string text4 = Path.Combine(sceneFilePath.GetBaseDir(), "layers", text3 + "_" + text);
		string path = text4 + "_a.tscn";
		string path2 = text4 + "_b.tscn";
		string path3 = text4 + "_c.tscn";
		if (ResourceLoader.Exists(path))
		{
			_layerA = ResourceLoader.Load<PackedScene>(path, null, ResourceLoader.CacheMode.Reuse);
		}
		if (ResourceLoader.Exists(path2))
		{
			_layerB = ResourceLoader.Load<PackedScene>(path2, null, ResourceLoader.CacheMode.Reuse);
		}
		if (ResourceLoader.Exists(path3))
		{
			_layerC = ResourceLoader.Load<PackedScene>(path3, null, ResourceLoader.CacheMode.Reuse);
		}
		UpdateLayers();
	}

	private void UpdateLayers()
	{
		ClearLayers();
		if (_visibleLayer == LayerVisibility.A && _layerA != null)
		{
			AddLayer(LayerVisibility.A, _layerA);
		}
		if (_visibleLayer == LayerVisibility.B && _layerB != null)
		{
			AddLayer(LayerVisibility.B, _layerB);
		}
		if (_visibleLayer == LayerVisibility.C && _layerC != null)
		{
			AddLayer(LayerVisibility.C, _layerC);
		}
	}

	private void AddLayer(LayerVisibility name, PackedScene layerScene)
	{
		Control control = layerScene.Instantiate<Control>(PackedScene.GenEditState.Disabled);
		control.Name = ToLayerName(name);
		this.AddChildSafely(control);
	}

	private static string ToLayerName(LayerVisibility layer)
	{
		return $"{"Layer_"}{layer}";
	}

	private IEnumerable<Control> GetLayerNodes()
	{
		foreach (Node child in GetChildren())
		{
			if (child.Name.ToString().StartsWith("Layer_"))
			{
				yield return (Control)child;
			}
		}
	}

	private void ClearLayers()
	{
		foreach (Control layerNode in GetLayerNodes())
		{
			this.RemoveChildSafely(layerNode);
			layerNode.QueueFreeSafely();
		}
	}

	public override void _ExitTree()
	{
		ClearLayers();
	}
}
