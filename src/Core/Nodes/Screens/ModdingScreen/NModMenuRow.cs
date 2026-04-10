using System;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens.ModdingScreen;

public partial class NModMenuRow : NClickableControl
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/modding/modding_screen_row");

	private const float _selectedAlpha = 0.25f;

	private Panel _selectionHighlight;

	private NTickbox _tickbox;

	private NModdingScreen _screen;

	private bool _isSelected;

	public Mod? Mod { get; private set; }

	public static NModMenuRow? Create(NModdingScreen screen, Mod mod)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NModMenuRow nModMenuRow = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NModMenuRow>(PackedScene.GenEditState.Disabled);
		nModMenuRow.Mod = mod;
		nModMenuRow._screen = screen;
		return nModMenuRow;
	}

	public override void _Ready()
	{
		if (Mod != null)
		{
			_selectionHighlight = GetNode<Panel>("SelectionHighlight");
			NTickbox node = GetNode<NTickbox>("Tickbox");
			MegaRichTextLabel node2 = GetNode<MegaRichTextLabel>("Title");
			TextureRect node3 = GetNode<TextureRect>("PlatformIcon");
			Panel selectionHighlight = _selectionHighlight;
			Color modulate = _selectionHighlight.Modulate;
			modulate.A = 0f;
			selectionHighlight.Modulate = modulate;
			node.IsTicked = !(SaveManager.Instance.SettingsSave.ModSettings?.IsModDisabled(Mod) ?? false);
			node.Connect(NTickbox.SignalName.Toggled, Callable.From<NTickbox>(OnTickboxToggled));
			node2.Text = Mod.manifest?.name ?? Mod.manifest?.id ?? "<null>";
			node3.Texture = GetPlatformIcon(Mod.modSource);
			node2.Modulate = (Mod.wasLoaded ? Colors.White : StsColors.gray);
			node3.Modulate = (Mod.wasLoaded ? Colors.White : StsColors.gray);
			ConnectSignals();
		}
	}

	protected override void OnFocus()
	{
		if (!_isSelected)
		{
			Panel selectionHighlight = _selectionHighlight;
			Color darkBlue = StsColors.darkBlue;
			darkBlue.A = 0.25f;
			selectionHighlight.Modulate = darkBlue;
		}
	}

	protected override void OnUnfocus()
	{
		if (!_isSelected)
		{
			_selectionHighlight.Modulate = Colors.Transparent;
		}
	}

	protected override void OnRelease()
	{
		_screen.OnRowSelected(this);
	}

	public void SetSelected(bool isSelected)
	{
		if (_isSelected != isSelected)
		{
			_isSelected = isSelected;
			if (_isSelected)
			{
				Panel selectionHighlight = _selectionHighlight;
				Color blue = StsColors.blue;
				blue.A = 0.25f;
				selectionHighlight.Modulate = blue;
			}
			else if (base.IsFocused)
			{
				Panel selectionHighlight2 = _selectionHighlight;
				Color blue = StsColors.darkBlue;
				blue.A = 0.25f;
				selectionHighlight2.Modulate = blue;
			}
			else
			{
				_selectionHighlight.Modulate = Colors.Transparent;
			}
		}
	}

	private void OnTickboxToggled(NTickbox tickbox)
	{
		SettingsSave settingsSave = SaveManager.Instance.SettingsSave;
		if (settingsSave.ModSettings == null)
		{
			ModSettings modSettings = (settingsSave.ModSettings = new ModSettings());
		}
		foreach (SettingsSaveMod mod in SaveManager.Instance.SettingsSave.ModSettings.ModList)
		{
			if (mod.Id == Mod?.manifest?.id)
			{
				mod.IsEnabled = tickbox.IsTicked;
			}
		}
		_screen.OnModEnabledOrDisabled();
	}

	public static Texture2D GetPlatformIcon(ModSource modSource)
	{
		AssetCache cache = PreloadManager.Cache;
		return cache.GetTexture2D(modSource switch
		{
			ModSource.ModsDirectory => ImageHelper.GetImagePath("ui/mods/folder.png"), 
			ModSource.SteamWorkshop => ImageHelper.GetImagePath("ui/mods/steam_logo.png"), 
			_ => throw new ArgumentOutOfRangeException("modSource", modSource, null), 
		});
	}
}
