using System;
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

	private bool _runningModded;

	public override void _Ready()
	{
		_releaseInfo = GetNode<MegaLabel>("%ReleaseInfo");
		_moddedWarning = GetNode<MegaLabel>("%ModdedWarning");
		_seed = GetNodeOrNull<MegaLabel>("%DebugSeed");
		_runningModded = ModManager.LoadedMods.Count > 0;
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
		string text = DateTime.Now.ToString("yyyy-MM-dd");
		string text2 = releaseInfo?.Version ?? commitId ?? "NONE";
		if (isMainMenu)
		{
			_releaseInfo.Text = text2 + "\n" + text;
		}
		else
		{
			_releaseInfo.Text = $"[{text2}] ({text})";
		}
		_moddedWarning.Visible = _runningModded;
		if (_runningModded)
		{
			bool flag = ModManager.LoadedMods.Any((Mod m) => !(m.assemblyLoadedSuccessfully ?? true));
			if (isMainMenu)
			{
				LocString locString = new LocString("main_menu_ui", "MODDED_WARNING");
				locString.Add("count", ModManager.LoadedMods.Count);
				locString.Add("hasError", flag);
				_moddedWarning.SetTextAutoSize(locString.GetFormattedText());
			}
			else
			{
				_moddedWarning.SetTextAutoSize($"MODDED ({ModManager.LoadedMods.Count})");
			}
			if (flag)
			{
				_moddedWarning.Modulate = StsColors.redGlow;
			}
		}
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (inputEvent.IsActionReleased(DebugHotkey.hideVersionInfo))
		{
			_releaseInfo.Visible = !_releaseInfo.Visible;
			_moddedWarning.Visible = _runningModded && !_moddedWarning.Visible;
			_seed?.SetVisible(!_seed.Visible);
			NGame.Instance.AddChildSafely(NFullscreenTextVfx.Create(_releaseInfo.Visible ? "Show Version Info" : "Hide Version Info"));
		}
	}
}
