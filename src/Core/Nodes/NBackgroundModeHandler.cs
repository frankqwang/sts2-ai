using System;
using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes;

public partial class NBackgroundModeHandler : Node
{
	private const int _backgroundFps = 30;

	private int _savedMaxFps;

	private bool _isBackgrounded;

	private static bool IsHeadless => DisplayServer.GetName().Equals("headless", StringComparison.OrdinalIgnoreCase);

	private static bool IsEditor => OS.HasFeature("editor");

	public override void _Notification(int what)
	{
		if (!IsHeadless && !IsEditor && !NonInteractiveMode.IsActive)
		{
			if ((long)what == 1005)
			{
				EnterBackgroundMode();
			}
			else if ((long)what == 1004)
			{
				ExitBackgroundMode();
			}
		}
	}

	private void EnterBackgroundMode()
	{
		if (!_isBackgrounded && SaveManager.Instance.SettingsSave.LimitFpsInBackground)
		{
			INetGameService netService = RunManager.Instance.NetService;
			if (netService == null || !netService.Type.IsMultiplayer())
			{
				_isBackgrounded = true;
				_savedMaxFps = Engine.MaxFps;
				Engine.MaxFps = 30;
				Log.Info($"Limiting background FPS to {30}");
			}
		}
	}

	private void ExitBackgroundMode()
	{
		if (_isBackgrounded)
		{
			_isBackgrounded = false;
			Engine.MaxFps = _savedMaxFps;
			Log.Info("Restored foreground FPS");
		}
	}
}
