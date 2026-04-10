using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;

namespace MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;

public partial class NPatchNotesScreen : Control, IScreenContext
{
	private NScrollableContainer _screenContents;

	private MarginContainer _marginContainer;

	private NButton _prevButton;

	private NButton _nextButton;

	private NButton _patchNotesToggle;

	private NButton _backButton;

	private MegaLabel _dateLabel;

	private MegaRichTextLabel _patchText;

	private Tween? _tween;

	private PackedScene _cachedScene;

	private const string _patchNotesPath = "res://localization/eng/patch_notes";

	private List<string>? _patchNotePaths;

	private int _index;

	private int _currentScrollLine;

	public bool IsOpen { get; private set; }

	public Control? DefaultFocusedControl => null;

	public override void _Ready()
	{
		_cachedScene = ResourceLoader.Load<PackedScene>("res://scenes/screens/patch_screen_contents.tscn", null, ResourceLoader.CacheMode.Reuse);
		CreateNewPatchEntry();
		_prevButton = GetNode<NButton>("PrevButton");
		_prevButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(delegate
		{
			PreviousPatchNote();
		}));
		_nextButton = GetNode<NButton>("NextButton");
		_nextButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(delegate
		{
			NextPatchNote();
		}));
		_nextButton.Visible = false;
		_patchNotesToggle = GetNode<NButton>("%PatchNotesToggle");
		_patchNotesToggle.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(delegate
		{
			Close();
		}));
		_patchNotesToggle.Disable();
		_backButton = GetNode<NButton>("%BackButton");
		_backButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(delegate
		{
			Close();
		}));
	}

	private void CreateNewPatchEntry()
	{
		_screenContents = _cachedScene.Instantiate<NScrollableContainer>(PackedScene.GenEditState.Disabled);
		this.AddChildSafely(_screenContents);
		MoveChild(_screenContents, 0);
		_marginContainer = _screenContents.GetNode<MarginContainer>("Content");
		_patchText = _screenContents.GetNode<MegaRichTextLabel>("Content/PatchText");
		_dateLabel = _patchText.GetNode<MegaLabel>("DateLabel");
		if (_patchNotePaths != null)
		{
			string patchNotePath = _patchNotePaths[_index];
			LoadPatchNoteText(patchNotePath);
		}
	}

	private void NextPatchNote()
	{
		if (!_nextButton.Visible)
		{
			return;
		}
		if (_patchNotePaths == null)
		{
			Log.Error("NPatchNotesScreen: No patch paths available!");
			return;
		}
		_index--;
		_prevButton.Visible = true;
		if (_index == 0)
		{
			_nextButton.Visible = false;
		}
		_screenContents.QueueFreeSafely();
		CreateNewPatchEntry();
	}

	private void PreviousPatchNote()
	{
		if (_patchNotePaths == null)
		{
			Log.Error("NPatchNotesScreen: No patch paths available!");
			return;
		}
		_index++;
		_nextButton.Visible = true;
		if (_index == _patchNotePaths.Count - 1)
		{
			_prevButton.Visible = false;
		}
		_screenContents.QueueFreeSafely();
		CreateNewPatchEntry();
	}

	public void Open()
	{
		IsOpen = true;
		NGame.Instance.MainMenu?.EnableBackstop();
		_patchNotesToggle.Enable();
		_backButton.Enable();
		base.Visible = true;
		_tween?.FastForwardToCompletion();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "modulate:a", 1f, 0.25);
		if (_patchNotePaths == null)
		{
			_patchNotePaths = (from fileName in DirAccess.GetFilesAt("res://localization/eng/patch_notes")
				select "res://localization/eng/patch_notes/" + fileName).Reverse().ToList();
		}
		LoadPatchNoteText(_patchNotePaths[_index]);
		ActiveScreenContext.Instance.Update();
		NHotkeyManager.Instance.PushHotkeyReleasedBinding(MegaInput.left, PreviousPatchNote);
		NHotkeyManager.Instance.PushHotkeyReleasedBinding(MegaInput.right, NextPatchNote);
		NHotkeyManager.Instance.PushHotkeyReleasedBinding(MegaInput.pauseAndBack, Close);
	}

	private void Close()
	{
		NHotkeyManager.Instance.RemoveHotkeyReleasedBinding(MegaInput.left, PreviousPatchNote);
		NHotkeyManager.Instance.RemoveHotkeyReleasedBinding(MegaInput.right, NextPatchNote);
		NHotkeyManager.Instance.RemoveHotkeyReleasedBinding(MegaInput.pauseAndBack, Close);
		_patchNotesToggle.Disable();
		_backButton.Disable();
		NGame.Instance.MainMenu?.DisableBackstop();
		_tween?.FastForwardToCompletion();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "modulate:a", 0f, 0.25);
		_tween.TweenCallback(Callable.From(delegate
		{
			IsOpen = false;
			SetVisible(visible: false);
			ActiveScreenContext.Instance.Update();
		}));
	}

	private void LoadPatchNoteText(string patchNotePath)
	{
		_patchText.ScrollToLine(0);
		_currentScrollLine = 0;
		string textAutoSize = ReadPatchNoteFile(patchNotePath);
		_patchText.SetTextAutoSize(textAutoSize);
		UpdateDateLabel(patchNotePath);
	}

	private static string ReadPatchNoteFile(string patchNotePath)
	{
		using FileAccess fileAccess = FileAccess.Open(patchNotePath, FileAccess.ModeFlags.Read);
		return fileAccess.GetAsText();
	}

	private void UpdateDateLabel(string patchNotePath)
	{
		string fileNameFromPath = GetFileNameFromPath(patchNotePath);
		string text = RemoveFileExtension(fileNameFromPath);
		if (TryParseDate(text, out string formattedDate))
		{
			_dateLabel.SetTextAutoSize(formattedDate);
		}
		else
		{
			Log.Error("Invalid date format in file name: " + text);
		}
	}

	private static string GetFileNameFromPath(string path)
	{
		int num = path.LastIndexOf('/') + 1;
		return path.Substring(num, path.Length - num);
	}

	private static string RemoveFileExtension(string fileName)
	{
		return fileName.Split('.')[0];
	}

	private static bool TryParseDate(string dateString, out string formattedDate)
	{
		if (DateTime.TryParseExact(dateString, "yyyy_MM_d", CultureInfo.InvariantCulture, DateTimeStyles.None, out var result))
		{
			formattedDate = result.ToString("MMMM d, yyyy", CultureInfo.InvariantCulture);
			return true;
		}
		formattedDate = string.Empty;
		return false;
	}
}
