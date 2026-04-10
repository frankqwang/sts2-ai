using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Platform;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NResolutionDropdown : NSettingsDropdown
{
	[Export(PropertyHint.None, "")]
	private PackedScene _dropdownItemScene;

	private Control _arrow;

	private static Vector2I _currentResolution;

	public static NResolutionDropdown? Instance { get; private set; }

	public override void _EnterTree()
	{
		Instance = this;
	}

	public override void _Ready()
	{
		ConnectSignals();
		_arrow = GetNode<Control>("%Arrow");
		NGame.Instance.Connect(NGame.SignalName.WindowChange, Callable.From<bool>(OnWindowChange));
		RefreshEnabled();
		RefreshCurrentlySelectedResolution();
		PopulateDropdownItems();
	}

	public void RefreshCurrentlySelectedResolution()
	{
		if (base.IsEnabled)
		{
			_currentResolution = DisplayServer.WindowGetSize();
			_currentOptionLabel.SetTextAutoSize($"{_currentResolution.X} x {_currentResolution.Y}");
		}
	}

	public void PopulateDropdownItems()
	{
		ClearDropdownItems();
		Vector2I boundaryResolution = DisplayServer.ScreenGetSize(SaveManager.Instance.SettingsSave.TargetDisplay);
		foreach (Vector2I resolutionWhite in GetResolutionWhiteList())
		{
			if (DoesResolutionFit(resolutionWhite, boundaryResolution))
			{
				NResolutionDropdownItem nResolutionDropdownItem = _dropdownItemScene.Instantiate<NResolutionDropdownItem>(PackedScene.GenEditState.Disabled);
				_dropdownItems.AddChildSafely(nResolutionDropdownItem);
				nResolutionDropdownItem.Connect(NDropdownItem.SignalName.Selected, Callable.From<NDropdownItem>(OnDropdownItemSelected));
				nResolutionDropdownItem.Init(resolutionWhite);
			}
		}
		_dropdownItems.GetParent<NDropdownContainer>().RefreshLayout();
	}

	private void OnWindowChange(bool isAutoAspectRatio)
	{
		RefreshEnabled();
		RefreshCurrentlySelectedResolution();
	}

	private void RefreshEnabled()
	{
		if (SaveManager.Instance.SettingsSave.Fullscreen || PlatformUtil.GetSupportedWindowMode().ShouldForceFullscreen())
		{
			Disable();
		}
		else
		{
			Enable();
		}
	}

	protected override void OnEnable()
	{
		_currentOptionLabel.Modulate = StsColors.gold;
		_arrow.Visible = true;
		RefreshCurrentlySelectedResolution();
	}

	protected override void OnDisable()
	{
		_currentOptionLabel.SetTextAutoSize("N/A");
		_currentOptionLabel.Modulate = StsColors.gray;
		_arrow.Visible = false;
	}

	private void OnDropdownItemSelected(NDropdownItem nDropdownItem)
	{
		NResolutionDropdownItem nResolutionDropdownItem = (NResolutionDropdownItem)nDropdownItem;
		if (!(nResolutionDropdownItem.resolution == _currentResolution))
		{
			CloseDropdown();
			SaveManager.Instance.SettingsSave.WindowPosition = DisplayServer.WindowGetPosition() - DisplayServer.ScreenGetPosition(SaveManager.Instance.SettingsSave.TargetDisplay);
			SaveManager.Instance.SettingsSave.WindowSize = nResolutionDropdownItem.resolution;
			Log.Info($"Setting window size to {nResolutionDropdownItem.resolution} from dropdown");
			NGame.Instance.ApplyDisplaySettings();
		}
	}

	private static bool DoesResolutionFit(Vector2I resolution, Vector2I boundaryResolution)
	{
		if (resolution.X <= boundaryResolution.X)
		{
			return resolution.Y <= boundaryResolution.Y;
		}
		return false;
	}

	private static List<Vector2I> GetResolutionWhiteList()
	{
		int num = 26;
		List<Vector2I> list = new List<Vector2I>(num);
		CollectionsMarshal.SetCount(list, num);
		Span<Vector2I> span = CollectionsMarshal.AsSpan(list);
		int num2 = 0;
		span[num2] = new Vector2I(1024, 768);
		num2++;
		span[num2] = new Vector2I(1152, 864);
		num2++;
		span[num2] = new Vector2I(1280, 720);
		num2++;
		span[num2] = new Vector2I(1280, 800);
		num2++;
		span[num2] = new Vector2I(1280, 960);
		num2++;
		span[num2] = new Vector2I(1366, 768);
		num2++;
		span[num2] = new Vector2I(1400, 1050);
		num2++;
		span[num2] = new Vector2I(1440, 900);
		num2++;
		span[num2] = new Vector2I(1440, 1080);
		num2++;
		span[num2] = new Vector2I(1600, 900);
		num2++;
		span[num2] = new Vector2I(1600, 1200);
		num2++;
		span[num2] = new Vector2I(1680, 1050);
		num2++;
		span[num2] = new Vector2I(1856, 1392);
		num2++;
		span[num2] = new Vector2I(1920, 1080);
		num2++;
		span[num2] = new Vector2I(1920, 1200);
		num2++;
		span[num2] = new Vector2I(1920, 1440);
		num2++;
		span[num2] = new Vector2I(2048, 1536);
		num2++;
		span[num2] = new Vector2I(2560, 1080);
		num2++;
		span[num2] = new Vector2I(2560, 1440);
		num2++;
		span[num2] = new Vector2I(2560, 1600);
		num2++;
		span[num2] = new Vector2I(3200, 1800);
		num2++;
		span[num2] = new Vector2I(3440, 1440);
		num2++;
		span[num2] = new Vector2I(3840, 1600);
		num2++;
		span[num2] = new Vector2I(3840, 2160);
		num2++;
		span[num2] = new Vector2I(3840, 2400);
		num2++;
		span[num2] = new Vector2I(7680, 4320);
		return list;
	}
}
