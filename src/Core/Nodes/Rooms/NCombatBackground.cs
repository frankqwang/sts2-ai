using System;
using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Rooms;

namespace MegaCrit.Sts2.Core.Nodes.Rooms;

public partial class NCombatBackground : Control
{
	public static NCombatBackground Create(BackgroundAssets bg)
	{
		if (bg.BackgroundScenePath == null)
		{
			throw new InvalidOperationException("Encounter does not have a background.");
		}
		string backgroundScenePath = bg.BackgroundScenePath;
		NCombatBackground nCombatBackground = PreloadManager.Cache.GetScene(backgroundScenePath).Instantiate<NCombatBackground>(PackedScene.GenEditState.Disabled);
		nCombatBackground.SetLayers(bg);
		return nCombatBackground;
	}

	private void SetLayers(BackgroundAssets bg)
	{
		SetBackgroundLayers(bg.BgLayers);
		SetForegroundLayer(bg.FgLayer);
	}

	private void SetBackgroundLayers(IReadOnlyList<string> backgroundLayers)
	{
		for (int i = 0; i < backgroundLayers.Count; i++)
		{
			string layerName = $"Layer_{i:D2}";
			AddLayer(layerName, backgroundLayers[i]);
		}
	}

	private void SetForegroundLayer(string? foregroundLayer)
	{
		if (foregroundLayer != null)
		{
			AddLayer("Foreground", foregroundLayer);
		}
	}

	private void AddLayer(string layerName, string layerPath)
	{
		Node nodeOrNull = GetNodeOrNull(layerName);
		if (nodeOrNull == null)
		{
			throw new InvalidOperationException("Layer node='" + layerName + "' not found in combat background scene.");
		}
		Control control = PreloadManager.Cache.GetScene(layerPath).Instantiate<Control>(PackedScene.GenEditState.Disabled);
		control.Visible = true;
		nodeOrNull.AddChildSafely(control);
	}
}
