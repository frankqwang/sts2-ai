using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Debug;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Nodes.Vfx;

namespace MegaCrit.Sts2.Core.Nodes.Debug;

public partial class NDebugInfoLabelManager : Node
{
	[Export(PropertyHint.None, "")]
	public bool isMainMenu;

	private MegaLabel _releaseInfo;

	private MegaLabel _moddedWarning;

	private MegaLabel? _seed;

	private Control? _modWarningContainer;

	private MegaRichTextLabel? _modWarningLabel;

	public override void _Ready()
	{
		_releaseInfo = GetNode<MegaLabel>("%ReleaseInfo");
		_moddedWarning = GetNode<MegaLabel>("%ModdedWarning");
		_seed = GetNodeOrNull<MegaLabel>("%DebugSeed");
		_modWarningContainer = GetNodeOrNull<Control>("%ModWarningContainer");
		_modWarningLabel = GetNodeOrNull<MegaRichTextLabel>("%ModWarningLabel");
		_moddedWarning.Connect(Control.SignalName.MouseEntered, Callable.From(OnModdedWarningHovered));
		_moddedWarning.Connect(Control.SignalName.MouseExited, Callable.From(OnModdedWarningUnhovered));
		_modWarningContainer?.SetVisible(visible: false);
		UpdateText(null);
		if (ReleaseInfoManager.Instance.ReleaseInfo == null)
		{
			TaskHelper.RunSafely(SetCommitIdInEditor());
		}
	}

	private async Task SetCommitIdInEditor()
	{
		if (GitHelper.ShortCommitIdTask != null)
		{
			UpdateText(await GitHelper.ShortCommitIdTask);
		}
	}

	private void UpdateText(string? commitId)
	{
		ReleaseInfo releaseInfo = ReleaseInfoManager.Instance.ReleaseInfo;
		string text = releaseInfo?.Date.ToString("yyyy.MM.dd") ?? "???";
		string text2 = releaseInfo?.Version ?? commitId ?? "NONE";
		if (isMainMenu)
		{
			_releaseInfo.Text = text2 + "\n" + text;
		}
		else
		{
			_releaseInfo.Text = $"[{text2}] ({text})";
		}
		_moddedWarning.Visible = ModManager.IsRunningModded();
		if (!_moddedWarning.Visible)
		{
			return;
		}
		bool flag = ModManager.Mods.Any(delegate(Mod m)
		{
			if (m.state != ModLoadState.Failed)
			{
				List<LocString>? errors = m.errors;
				if (errors == null)
				{
					return false;
				}
				return errors.Count > 0;
			}
			return true;
		});
		if (isMainMenu)
		{
			LocString locString = new LocString("main_menu_ui", "MODDED_WARNING");
			locString.Add("count", ModManager.GetLoadedMods().Count());
			locString.Add("hasError", flag);
			_moddedWarning.SetTextAutoSize(locString.GetFormattedText());
			LocString[] array = ModManager.Mods.SelectMany((Mod m) => m.errors ?? new List<LocString>()).ToArray();
			if (array.Length != 0)
			{
				_modWarningLabel.Text = string.Join("\n", array.Select((LocString s) => s.GetFormattedText()));
			}
			else
			{
				LocString locString2 = new LocString("main_menu_ui", "MOD_ERROR.NONE");
				locString2.Add("mods", string.Join(", ", ModManager.GetLoadedMods().Select(delegate(Mod m)
				{
					object obj = m.manifest?.name;
					if (obj == null)
					{
						ModManifest? manifest = m.manifest;
						if (manifest == null)
						{
							return (string)null;
						}
						obj = manifest.id;
					}
					return (string)obj;
				})));
				_modWarningLabel.Text = locString2.GetFormattedText();
			}
		}
		else
		{
			_moddedWarning.SetTextAutoSize($"MODDED ({ModManager.GetLoadedMods().Count()})");
		}
		if (flag)
		{
			_moddedWarning.Modulate = StsColors.redGlow;
		}
	}

	private void OnModdedWarningHovered()
	{
		if (_modWarningContainer != null)
		{
			_modWarningContainer.Visible = true;
		}
	}

	private void OnModdedWarningUnhovered()
	{
		if (_modWarningContainer != null)
		{
			_modWarningContainer.Visible = false;
		}
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (inputEvent.IsActionReleased(DebugHotkey.hideVersionInfo))
		{
			_releaseInfo.Visible = !_releaseInfo.Visible;
			_moddedWarning.Visible = ModManager.IsRunningModded() && !_moddedWarning.Visible;
			_seed?.SetVisible(!_seed.Visible);
			NGame.Instance.AddChildSafely(NFullscreenTextVfx.Create(_releaseInfo.Visible ? "Show Version Info" : "Hide Version Info"));
		}
	}
}
