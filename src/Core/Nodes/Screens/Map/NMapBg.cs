using System;
using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Map;

public partial class NMapBg : VBoxContainer
{
	private IRunState _runState;

	private TextureRect _mapTop;

	private TextureRect _mapMid;

	private TextureRect _mapBot;

	private NMapDrawings _drawings;

	private Window _window;

	private const float _sixteenByNine = 1.7777778f;

	private const float _fourByThree = 1.3333334f;

	private const float _defaultY = -1620f;

	private const float _adjustY = -1540f;

	private float _offsetX;

	public override void _Ready()
	{
		_mapTop = GetNode<TextureRect>("MapTop");
		_mapMid = GetNode<TextureRect>("MapMid");
		_mapBot = GetNode<TextureRect>("MapBot");
		_drawings = GetNode<NMapDrawings>("%Drawings");
		_window = GetTree().Root;
		_window.Connect(Viewport.SignalName.SizeChanged, Callable.From(OnWindowChange));
		OnWindowChange();
		_offsetX = base.Position.X;
		Connect(CanvasItem.SignalName.VisibilityChanged, Callable.From(OnVisibilityChanged));
	}

	public void Initialize(IRunState runState)
	{
		_runState = runState;
	}

	private void OnVisibilityChanged()
	{
		ActModel act = _runState.Act;
		_mapTop.Texture = act.MapTopBg;
		_mapMid.Texture = act.MapMidBg;
		_mapBot.Texture = act.MapBotBg;
	}

	private void OnWindowChange()
	{
		float num = Math.Max(1.3333334f, (float)_window.Size.X / (float)_window.Size.Y);
		if (num < 1.7777778f)
		{
			float p = (num - 1.3333334f) / 0.44444442f;
			base.Position = new Vector2(_offsetX, Mathf.Remap(Ease.CubicOut(p), 0f, 1f, -1540f, -1620f));
		}
		else
		{
			base.Position = new Vector2(_offsetX, -1620f);
		}
		_drawings.RepositionBasedOnBackground(this);
	}
}
