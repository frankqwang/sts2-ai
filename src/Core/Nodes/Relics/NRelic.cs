using System;
using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Relics;

public partial class NRelic : Control
{
	public enum IconSize
	{
		Small,
		Large
	}

	public const string relicMatPath = "res://materials/ui/relic_mat.tres";

	private static readonly string _scenePath = SceneHelper.GetScenePath("relics/relic");

	private RelicModel? _model;

	private IconSize _iconSize;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlyArray<string>(new string[2] { "res://materials/ui/relic_mat.tres", _scenePath });

	public TextureRect Icon { get; private set; }

	public TextureRect Outline { get; private set; }

	public RelicModel Model
	{
		get
		{
			return _model ?? throw new InvalidOperationException("Model was accessed before it was set.");
		}
		set
		{
			if (_model != value)
			{
				RelicModel model = _model;
				_model = value;
				this.ModelChanged?.Invoke(model, _model);
			}
			Reload();
		}
	}

	public event Action<RelicModel?, RelicModel?>? ModelChanged;

	public static NRelic? Create(RelicModel relic, IconSize iconSize)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NRelic nRelic = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NRelic>(PackedScene.GenEditState.Disabled);
		nRelic.Name = $"NRelic-{relic.Id}";
		nRelic.Model = relic;
		nRelic._iconSize = iconSize;
		return nRelic;
	}

	public override void _Ready()
	{
		Icon = GetNode<TextureRect>("%Icon");
		Outline = GetNode<TextureRect>("%Outline");
		Reload();
	}

	private void Reload()
	{
		if (IsNodeReady() && _model != null)
		{
			Model.UpdateTexture(Icon);
			switch (_iconSize)
			{
			case IconSize.Small:
				Icon.Texture = Model.Icon;
				Outline.Visible = true;
				Outline.Texture = Model.IconOutline;
				break;
			case IconSize.Large:
				Icon.Texture = Model.BigIcon;
				Outline.Visible = false;
				break;
			default:
				throw new ArgumentOutOfRangeException("_iconSize", _iconSize, null);
			}
		}
	}
}
