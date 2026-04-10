using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.ProfileScreen;

public partial class NProfileButton : NButton
{
	private static readonly StringName _v = new StringName("v");

	private MegaRichTextLabel _title;

	private MegaRichTextLabel _description;

	private Control _currentProfileIndicator;

	private NProfileIcon _profileIcon;

	private NDeleteProfileButton _deleteButton;

	private TextureRect _background;

	private ShaderMaterial _hsv;

	private NProfileScreen? _profileScreen;

	private Tween? _tween;

	private int _profileId;

	public static IEnumerable<string> AssetPaths => NProfileIcon.AssetPaths;

	public override void _Ready()
	{
		ConnectSignals();
		_background = GetNode<TextureRect>("%Background");
		_hsv = (ShaderMaterial)_background.Material;
		_title = GetNode<MegaRichTextLabel>("%Title");
		_description = GetNode<MegaRichTextLabel>("%Info");
		_profileIcon = GetNode<NProfileIcon>("%ProfileIcon");
		_currentProfileIndicator = GetNode<Control>("%CurrentProfileIndicator");
	}

	public void Initialize(NProfileScreen profileScreen, int profileId)
	{
		_profileScreen = profileScreen;
		_profileId = profileId;
		LocString locString = new LocString("main_menu_ui", "PROFILE_SCREEN.BUTTON.title");
		locString.Add("Id", profileId);
		_title.Text = locString.GetFormattedText();
		GodotFileIo godotFileIo = new GodotFileIo(UserDataPathProvider.GetProfileScopedPath(profileId, "saves"));
		_profileIcon.SetProfileId(profileId);
		_currentProfileIndicator.Visible = SaveManager.Instance.CurrentProfileId == profileId;
		string path = "progress.save";
		if (NProfileScreen.forceShowProfileAsDeleted == profileId || !godotFileIo.FileExists(path))
		{
			LocString locString2 = new LocString("main_menu_ui", "PROFILE_SCREEN.BUTTON.empty");
			_description.Text = locString2.GetFormattedText();
			return;
		}
		LocString locString3 = new LocString("main_menu_ui", "PROFILE_SCREEN.BUTTON.description");
		if (SaveManager.Instance.CurrentProfileId == profileId)
		{
			locString3.Add("Playtime", TimeFormatting.Format(SaveManager.Instance.Progress.TotalPlaytime));
		}
		else
		{
			string json = godotFileIo.ReadFile(path);
			JsonObject jsonObject = JsonSerializer.Deserialize(json, JsonSerializationUtility.GetTypeInfo<JsonObject>());
			if (jsonObject.TryGetPropertyValue("total_playtime", out JsonNode jsonNode) && jsonNode is JsonValue jsonValue && jsonValue.TryGetValue<long>(out var value))
			{
				locString3.Add("Playtime", TimeFormatting.Format(value));
			}
			else
			{
				locString3.Add("Playtime", "???");
			}
		}
		DateTimeOffset dateTimeOffset = godotFileIo.GetLastModifiedTime(path);
		string path2 = "current_run.save";
		if (godotFileIo.FileExists(path2))
		{
			DateTimeOffset lastModifiedTime = godotFileIo.GetLastModifiedTime(path2);
			dateTimeOffset = ((dateTimeOffset > lastModifiedTime) ? dateTimeOffset : lastModifiedTime);
		}
		string path3 = "current_run_mp.save";
		if (godotFileIo.FileExists(path3))
		{
			DateTimeOffset lastModifiedTime2 = godotFileIo.GetLastModifiedTime(path3);
			dateTimeOffset = ((dateTimeOffset > lastModifiedTime2) ? dateTimeOffset : lastModifiedTime2);
		}
		DateTimeFormatInfo dateTimeFormat = LocManager.Instance.CultureInfo.DateTimeFormat;
		DateTime dateTime = TimeZoneInfo.ConvertTimeFromUtc(dateTimeOffset.UtcDateTime, TimeZoneInfo.Local);
		LocString locString4 = new LocString("main_menu_ui", "PROFILE_SCREEN.BUTTON.dateFormat");
		string variable = dateTime.ToString(locString4.GetFormattedText(), dateTimeFormat);
		locString3.Add("LastUpdatedTime", variable);
		_description.Text = locString3.GetFormattedText();
	}

	protected override void OnRelease()
	{
		if (SaveManager.Instance.CurrentProfileId == _profileId)
		{
			NGame.Instance.MainMenu.SubmenuStack.Pop();
		}
		else
		{
			TaskHelper.RunSafely(SwitchToThisProfile());
		}
	}

	private async Task SwitchToThisProfile()
	{
		_profileScreen?.ShowLoading();
		await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		SaveManager.Instance.SwitchProfileId(_profileId);
		ReadSaveResult<PrefsSave> prefsReadResult = SaveManager.Instance.InitPrefsData();
		ReadSaveResult<SerializableProgress> progressReadResult = SaveManager.Instance.InitProgressData();
		NGame.Instance.ReloadMainMenu();
		NGame.Instance.CheckShowSaveFileError(progressReadResult, prefsReadResult, new ReadSaveResult<SettingsSave>(new SettingsSave()));
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "scale", Vector2.One * 1.03f, 0.05);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 1.3f, 0.05);
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	private void UpdateShaderV(float value)
	{
		_hsv.SetShaderParameter(_v, value);
	}
}
