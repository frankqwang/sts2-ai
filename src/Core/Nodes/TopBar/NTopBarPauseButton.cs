using System;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.Capstones;
using MegaCrit.Sts2.Core.Nodes.Screens.PauseMenu;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.TopBar;

public partial class NTopBarPauseButton : NTopBarButton
{
	private static readonly StringName _v = new StringName("v");

	private static readonly HoverTip _hoverTip = new HoverTip(new LocString("static_hover_tips", "SETTINGS.title"), new LocString("static_hover_tips", "SETTINGS.description"));

	private const float _hoverAngle = -(float)Math.PI;

	private const float _hoverShaderV = 1.1f;

	private const float _defaultV = 0.9f;

	private const float _pressDownV = 0.4f;

	private IRunState _runState;

	protected override string[] Hotkeys => new string[1] { MegaInput.pauseAndBack };

	protected override void OnRelease()
	{
		base.OnRelease();
		if (IsOpen())
		{
			NCapstoneContainer.Instance.Close();
		}
		else
		{
			NPauseMenu nPauseMenu = (NPauseMenu)NRun.Instance.GlobalUi.SubmenuStack.ShowScreen(CapstoneSubmenuType.PauseMenu);
			nPauseMenu.Initialize(_runState);
		}
		UpdateScreenOpen();
		_hsv?.SetShaderParameter(_v, 0.9f);
	}

	protected override bool IsOpen()
	{
		if (NCapstoneContainer.Instance.CurrentCapstoneScreen is NCapstoneSubmenuStack nCapstoneSubmenuStack)
		{
			return nCapstoneSubmenuStack.ScreenType == NetScreenType.PauseMenu;
		}
		return false;
	}

	public override void _Process(double delta)
	{
		if (base.IsScreenOpen)
		{
			_icon.Rotation += (float)delta;
		}
	}

	public void Initialize(IRunState runState)
	{
		_runState = runState;
	}

	protected override async Task AnimPressDown(CancellationTokenSource cancelToken)
	{
		float timer = 0f;
		float startAngle = _icon.Rotation;
		float targetAngle = startAngle + (float)Math.PI / 4f;
		for (; timer < 0.25f; timer += (float)GetProcessDeltaTime())
		{
			if (cancelToken.IsCancellationRequested)
			{
				return;
			}
			_icon.Rotation = Mathf.LerpAngle(startAngle, targetAngle, Ease.CubicOut(timer / 0.25f));
			_hsv?.SetShaderParameter(_v, Mathf.Lerp(1.1f, 0.4f, Ease.CubicOut(timer / 0.25f)));
			if (!this.IsValid() || !IsInsideTree())
			{
				return;
			}
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		}
		_icon.Rotation = targetAngle;
		_hsv?.SetShaderParameter(_v, 0.4f);
	}

	protected override async Task AnimHover(CancellationTokenSource cancelToken)
	{
		float timer = 0f;
		float startAngle = _icon.Rotation;
		for (; timer < 0.5f; timer += (float)GetProcessDeltaTime())
		{
			if (cancelToken.IsCancellationRequested)
			{
				return;
			}
			_icon.Rotation = Mathf.LerpAngle(startAngle, -(float)Math.PI, Ease.BackOut(timer / 0.5f));
			if (!this.IsValid() || !IsInsideTree())
			{
				return;
			}
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		}
		_icon.Rotation = -(float)Math.PI;
	}

	protected override async Task AnimUnhover(CancellationTokenSource cancelToken)
	{
		float timer = 0f;
		float startAngle = _icon.Rotation;
		for (; timer < 1f; timer += (float)GetProcessDeltaTime())
		{
			if (cancelToken.IsCancellationRequested)
			{
				return;
			}
			_icon.Rotation = Mathf.LerpAngle(startAngle, 0f, Ease.ElasticOut(timer / 1f));
			_hsv?.SetShaderParameter(_v, Mathf.Lerp(1.1f, 0.9f, Ease.ExpoOut(timer / 1f)));
			_icon.Scale = NTopBarButton._hoverScale.Lerp(Vector2.One, Ease.ExpoOut(timer / 1f));
			if (!this.IsValid() || !IsInsideTree())
			{
				return;
			}
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		}
		_hsv?.SetShaderParameter(_v, 0.9f);
		_icon.Rotation = 0f;
		_icon.Scale = Vector2.One;
	}

	public void ToggleAnimState()
	{
		UpdateScreenOpen();
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, _hoverTip);
		nHoverTipSet.GlobalPosition = base.GlobalPosition + new Vector2(base.Size.X - nHoverTipSet.Size.X, base.Size.Y + 20f);
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		NHoverTipSet.Remove(this);
	}
}
