using System;
using System.Globalization;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Saves.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;

public partial class NContinueRunInfo : Control
{
	private Tween? _visTween;

	private Vector2 _initPosition;

	private Control _runInfoContainer;

	private Control _errorContainer;

	private MegaRichTextLabel _dateLabel;

	private MegaRichTextLabel _goldLabel;

	private MegaRichTextLabel _healthLabel;

	private MegaRichTextLabel _progressLabel;

	private MegaRichTextLabel _ascensionLabel;

	private TextureRect _charIcon;

	private bool _isShown;

	public bool HasResult { get; private set; }

	public override void _Ready()
	{
		_initPosition = base.Position;
		base.Modulate = StsColors.transparentWhite;
		_runInfoContainer = GetNode<VBoxContainer>("%RunInfoContainer");
		_errorContainer = GetNode<Control>("%ErrorContainer");
		_dateLabel = GetNode<MegaRichTextLabel>("%DateLabel");
		_goldLabel = GetNode<MegaRichTextLabel>("%GoldLabel");
		_healthLabel = GetNode<MegaRichTextLabel>("%HealthLabel");
		_progressLabel = GetNode<MegaRichTextLabel>("%ProgressLabel");
		_charIcon = GetNode<TextureRect>("%CharacterIcon");
		_ascensionLabel = GetNode<MegaRichTextLabel>("%AscensionLabel");
	}

	public void AnimShow()
	{
		_visTween?.Kill();
		_visTween = CreateTween().SetParallel();
		_visTween.TweenProperty(this, "position", _initPosition + new Vector2(0f, -20f), 0.20000000298023224).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back);
		_visTween.TweenProperty(this, "modulate:a", 1f, 0.20000000298023224);
		_isShown = true;
	}

	public void AnimHide()
	{
		if (_isShown)
		{
			_visTween?.Kill();
			_visTween = CreateTween().SetParallel();
			_visTween.TweenProperty(this, "position", _initPosition, 0.20000000298023224);
			_visTween.TweenProperty(this, "modulate:a", 0f, 0.20000000298023224);
			_isShown = false;
		}
	}

	public void SetResult(ReadSaveResult<SerializableRun>? result)
	{
		if (result != null && result.Success && result.SaveData != null)
		{
			ShowInfo(result.SaveData);
		}
		else if (result != null)
		{
			ShowError();
		}
		HasResult = result != null;
	}

	private void ShowInfo(SerializableRun save)
	{
		_errorContainer.Visible = false;
		_runInfoContainer.Visible = true;
		DateTimeFormatInfo dateTimeFormat = LocManager.Instance.CultureInfo.DateTimeFormat;
		DateTime dateTime = TimeZoneInfo.ConvertTimeFromUtc(DateTimeOffset.FromUnixTimeSeconds(save.SaveTime).UtcDateTime, TimeZoneInfo.Local);
		string rawText = new LocString("main_menu_ui", "CONTINUE_RUN_INFO.savedTimeFormat").GetRawText();
		string variable = dateTime.ToString(rawText, dateTimeFormat);
		LocString locString = new LocString("main_menu_ui", "CONTINUE_RUN_INFO.saved");
		locString.Add("LastSavedTime", variable);
		_dateLabel.Text = locString.GetFormattedText();
		if (save.Ascension > 0)
		{
			LocString locString2 = new LocString("main_menu_ui", "CONTINUE_RUN_INFO.ascension");
			_ascensionLabel.Text = $"{locString2.GetFormattedText()} {save.Ascension}";
		}
		else
		{
			_ascensionLabel.Visible = false;
		}
		ActModel byId = ModelDb.GetById<ActModel>(save.Acts[save.CurrentActIndex].Id);
		SerializablePlayer serializablePlayer = save.Players[0];
		_charIcon.Texture = ModelDb.GetById<CharacterModel>(serializablePlayer.CharacterId).IconTexture;
		string formattedText = byId.Title.GetFormattedText();
		string formattedText2 = new LocString("main_menu_ui", "CONTINUE_RUN_INFO.floor").GetFormattedText();
		int num = save.VisitedMapCoords.Count;
		for (int i = 0; i < save.CurrentActIndex; i++)
		{
			num += ModelDb.GetById<ActModel>(save.Acts[i].Id).GetNumberOfFloors(save.Players.Count > 1);
		}
		_progressLabel.Text = $"{formattedText} [blue]- {formattedText2} {num}[/blue]";
		_healthLabel.Text = $"[red]{serializablePlayer.CurrentHp}/{serializablePlayer.MaxHp}[/red]";
		_goldLabel.Text = $"[gold]{serializablePlayer.Gold}[/gold]";
	}

	private void ShowError()
	{
		_runInfoContainer.Visible = false;
		_errorContainer.Visible = true;
	}
}
