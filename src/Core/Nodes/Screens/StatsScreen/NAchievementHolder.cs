using System;
using System.Globalization;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Achievements;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Platform;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens.StatsScreen;

public partial class NAchievementHolder : Control
{
	private static readonly StringName _v = new StringName("v");

	private static readonly StringName _s = new StringName("s");

	private static readonly StringName _h = new StringName("h");

	private const string _scenePath = "screens/stats_screen/achievement_holder";

	private TextureRect _border;

	private TextureRect _icon;

	private TextureRect _lock;

	private ShaderMaterial _iconHsv;

	private ShaderMaterial _borderHsv;

	private MegaRichTextLabel _infoLabel;

	private MegaLabel _date;

	private Achievement _achievement;

	private Tween? _tween;

	public bool IsUnlocked { get; private set; }

	public static NAchievementHolder? Create(Achievement achievement)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NAchievementHolder nAchievementHolder = PreloadManager.Cache.GetScene(SceneHelper.GetScenePath("screens/stats_screen/achievement_holder")).Instantiate<NAchievementHolder>(PackedScene.GenEditState.Disabled);
		nAchievementHolder._achievement = achievement;
		nAchievementHolder.IsUnlocked = AchievementsUtil.IsUnlocked(achievement);
		return nAchievementHolder;
	}

	public override void _Ready()
	{
		_icon = GetNode<TextureRect>("%Icon");
		_border = GetNode<TextureRect>("%Border");
		_lock = GetNode<TextureRect>("%Lock");
		_borderHsv = (ShaderMaterial)_border.Material;
		_iconHsv = (ShaderMaterial)_icon.Material;
		_infoLabel = GetNode<MegaRichTextLabel>("%InfoText");
		_date = GetNode<MegaLabel>("%DateText");
		RefreshUnlocked();
	}

	public static string GetPathForAchievement(Enum achievement)
	{
		string text = StringHelper.SnakeCase(achievement.ToString()).ToLowerInvariant();
		return ImageHelper.GetImagePath("packed/achievements/unlocked/" + text + ".png");
	}

	public void RefreshUnlocked()
	{
		IsUnlocked = AchievementsUtil.IsUnlocked(_achievement);
		string text = StringHelper.SnakeCase(_achievement.ToString()).ToLowerInvariant();
		_icon.Texture = PreloadManager.Cache.GetTexture2D(GetPathForAchievement(_achievement));
		text = text.ToUpperInvariant();
		if (IsUnlocked)
		{
			_infoLabel.Text = "[b][gold]" + new LocString("achievements", text + ".title").GetRawText() + "[/gold][/b]\n" + new LocString("achievements", text + ".description").GetFormattedText();
		}
		else
		{
			_infoLabel.Text = "[b][red]" + new LocString("achievements", "LOCKED.title").GetRawText() + "[/red][/b]\n" + new LocString("achievements", text + ".description").GetFormattedText();
		}
		SetLockVisuals();
		SetDateLabel();
	}

	private void SetLockVisuals()
	{
		_lock.Visible = !IsUnlocked;
		if (IsUnlocked)
		{
			_borderHsv.SetShaderParameter(_h, 1f);
			_borderHsv.SetShaderParameter(_s, 1f);
			_borderHsv.SetShaderParameter(_v, 1f);
			_iconHsv.SetShaderParameter(_s, 1f);
			_iconHsv.SetShaderParameter(_v, 1f);
		}
		else
		{
			_borderHsv.SetShaderParameter(_h, 0.4f);
			_borderHsv.SetShaderParameter(_s, 0.4f);
			_borderHsv.SetShaderParameter(_v, 0.8f);
			_iconHsv.SetShaderParameter(_s, 0.2f);
			_iconHsv.SetShaderParameter(_v, 0.5f);
		}
	}

	private void SetDateLabel()
	{
		_date.Visible = IsUnlocked;
		if (IsUnlocked)
		{
			if (!SaveManager.Instance.Progress.UnlockedAchievements.TryGetValue(_achievement, out var value))
			{
				_date.Visible = false;
				return;
			}
			DateTimeFormatInfo dateTimeFormat = LocManager.Instance.CultureInfo.DateTimeFormat;
			DateTime dateTime = TimeZoneInfo.ConvertTimeFromUtc(DateTimeOffset.FromUnixTimeSeconds(value).UtcDateTime, TimeZoneInfo.Local);
			LocString locString = new LocString("achievements", "UNLOCK_DATE.text");
			string variable = dateTime.ToString(new LocString("achievements", "UNLOCK_DATE.format").GetRawText(), dateTimeFormat);
			locString.Add("Date", variable);
			_date.SetTextAutoSize(locString.GetFormattedText());
		}
	}
}
