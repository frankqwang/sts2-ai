using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Credits;

public partial class NCreditsScreen : Control, IScreenContext
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/credits_screen");

	private bool _canClose;

	private bool _exitingScreen;

	private Tween? _tween;

	private Control _screenContents;

	private NBackButton _backButton;

	private const string _table = "credits";

	private float _targetPosition;

	private const float _scrollSpeed = 50f;

	private const float _trackpadScrollSpeed = 20f;

	private const float _autoScrollSpeed = 80f;

	private const float _lerpSmoothness = 20f;

	public Control DefaultFocusedControl => this;

	public static NCreditsScreen? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NCreditsScreen>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		NHotkeyManager.Instance.PushHotkeyReleasedBinding(MegaInput.back, CloseScreenDebug);
		_screenContents = GetNode<Control>("%ScreenContents");
		_backButton = GetNode<NBackButton>("BackButton");
		_screenContents.Modulate = StsColors.transparentWhite;
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_screenContents, "modulate", Colors.White, 2.0);
		_targetPosition = _screenContents.Position.Y;
		TaskHelper.RunSafely(EnableScreenExit());
		InitMegaCrit();
		InitComposer();
		InitAdditionalProgramming();
		InitAdditionalVfx();
		InitMarketingSupport();
		InitConsultants();
		InitVoices();
		InitLocalization();
		InitTwitchExtension();
		InitModdingSupport();
		InitPlaytesters();
		InitTrailer();
		InitFmod();
		InitSpine();
		InitGodot();
		InitExitMessage();
	}

	private void InitMegaCrit()
	{
		GetNode<MegaLabel>("%MegaCritHeader").Text = new LocString("credits", "MEGA_CRIT.header").GetRawText();
		GetNode<MegaRichTextLabel>("%CreatedByNames").Text = new LocString("credits", "MEGA_CRIT.names").GetRawText();
		var (text, text2) = SplitTwoColumn(new LocString("credits", "MEGA_CRIT_TEAM.names").GetRawText());
		GetNode<MegaRichTextLabel>("%MegaCritTeamRoles").Text = text;
		GetNode<MegaRichTextLabel>("%MegaCritTeamNames").Text = text2;
	}

	private void InitComposer()
	{
		GetNode<MegaLabel>("%ComposerHeader").Text = new LocString("credits", "COMPOSER.header").GetRawText();
		GetNode<MegaRichTextLabel>("%ComposerNames").Text = new LocString("credits", "COMPOSER.names").GetRawText();
	}

	private void InitAdditionalProgramming()
	{
		GetNode<MegaLabel>("%AdditionalProgrammingHeader").Text = new LocString("credits", "ADDITIONAL_PROGRAMMING.header").GetRawText();
		GetNode<MegaRichTextLabel>("%AdditionalProgrammingNames").Text = new LocString("credits", "ADDITIONAL_PROGRAMMING.names").GetRawText();
	}

	private void InitAdditionalVfx()
	{
		GetNode<MegaLabel>("%AdditionalVfxHeader").Text = new LocString("credits", "ADDITIONAL_VFX.header").GetRawText();
		GetNode<MegaRichTextLabel>("%AdditionalVfxNames").Text = new LocString("credits", "ADDITIONAL_VFX.names").GetRawText();
	}

	private void InitMarketingSupport()
	{
		GetNode<MegaLabel>("%MarketingSupportHeader").Text = new LocString("credits", "MARKETING_SUPPORT.header").GetRawText();
		GetNode<MegaRichTextLabel>("%MarketingSupportNames").Text = new LocString("credits", "MARKETING_SUPPORT.names").GetRawText();
	}

	private void InitConsultants()
	{
		GetNode<MegaLabel>("%ConsultantsHeader").Text = new LocString("credits", "CONSULTANTS.header").GetRawText();
		GetNode<MegaRichTextLabel>("%ConsultantsNames").Text = new LocString("credits", "CONSULTANTS.names").GetRawText();
	}

	private void InitVoices()
	{
		GetNode<MegaLabel>("%VoicesHeader").Text = new LocString("credits", "VOICES.header").GetRawText();
		var (text, text2) = SplitTwoColumnMultiRole(new LocString("credits", "VOICES.names").GetRawText());
		GetNode<MegaRichTextLabel>("%VoicesRoles").Text = text;
		GetNode<MegaRichTextLabel>("%VoicesNames").Text = text2;
	}

	private void InitLocalization()
	{
		GetNode<MegaLabel>("%LocalizationHeader").Text = new LocString("credits", "LOC.header").GetRawText();
		GetNode<MegaLabel>("%ptbHeader").Text = new LocString("credits", "LOC_PTB.header").GetRawText();
		GetNode<MegaRichTextLabel>("%ptbNames").Text = new LocString("credits", "LOC_PTB.names").GetRawText();
		GetNode<MegaLabel>("%zhsHeader").Text = new LocString("credits", "LOC_ZHS.header").GetRawText();
		GetNode<MegaRichTextLabel>("%zhsNames").Text = new LocString("credits", "LOC_ZHS.names").GetRawText();
		GetNode<MegaLabel>("%fraHeader").Text = new LocString("credits", "LOC_FRA.header").GetRawText();
		(string, string) tuple = SplitTwoColumn(new LocString("credits", "LOC_FRA.names").GetRawText());
		GetNode<MegaRichTextLabel>("%fraRoles").Text = tuple.Item1;
		GetNode<MegaRichTextLabel>("%fraNames").Text = tuple.Item2;
		GetNode<MegaLabel>("%deuHeader").Text = new LocString("credits", "LOC_DEU.header").GetRawText();
		GetNode<MegaRichTextLabel>("%deuNames").Text = new LocString("credits", "LOC_DEU.names").GetRawText();
		GetNode<MegaLabel>("%itaHeader").Text = new LocString("credits", "LOC_ITA.header").GetRawText();
		GetNode<MegaRichTextLabel>("%itaTeam").Text = new LocString("credits", "LOC_ITA.team").GetRawText();
		tuple = SplitTwoColumn(new LocString("credits", "LOC_ITA.names").GetRawText());
		GetNode<MegaRichTextLabel>("%itaRoles").Text = tuple.Item1;
		GetNode<MegaRichTextLabel>("%itaNames").Text = tuple.Item2;
		GetNode<MegaLabel>("%jpnHeader").Text = new LocString("credits", "LOC_JPN.header").GetRawText();
		GetNode<MegaRichTextLabel>("%jpnNames").Text = new LocString("credits", "LOC_JPN.names").GetRawText();
		GetNode<MegaLabel>("%korHeader").Text = new LocString("credits", "LOC_KOR.header").GetRawText();
		GetNode<MegaRichTextLabel>("%korNames").Text = new LocString("credits", "LOC_KOR.names").GetRawText();
		GetNode<MegaLabel>("%polHeader").Text = new LocString("credits", "LOC_POL.header").GetRawText();
		GetNode<MegaRichTextLabel>("%polNames").Text = new LocString("credits", "LOC_POL.names").GetRawText();
		GetNode<MegaLabel>("%rusHeader").Text = new LocString("credits", "LOC_RUS.header").GetRawText();
		tuple = SplitTwoColumn(new LocString("credits", "LOC_RUS.names").GetRawText());
		GetNode<MegaRichTextLabel>("%rusRoles").Text = tuple.Item1;
		GetNode<MegaRichTextLabel>("%rusNames").Text = tuple.Item2;
		GetNode<MegaLabel>("%spaHeader").Text = new LocString("credits", "LOC_SPA.header").GetRawText();
		GetNode<MegaRichTextLabel>("%spaNames").Text = new LocString("credits", "LOC_SPA.names").GetRawText();
		GetNode<MegaLabel>("%espHeader").Text = new LocString("credits", "LOC_ESP.header").GetRawText();
		GetNode<MegaRichTextLabel>("%espTeam").Text = new LocString("credits", "LOC_ESP.team").GetRawText();
		tuple = SplitTwoColumn(new LocString("credits", "LOC_ESP.names").GetRawText());
		GetNode<MegaRichTextLabel>("%espRoles").Text = tuple.Item1;
		GetNode<MegaRichTextLabel>("%espNames").Text = tuple.Item2;
		GetNode<MegaLabel>("%thaHeader").Text = new LocString("credits", "LOC_THA.header").GetRawText();
		GetNode<MegaRichTextLabel>("%thaNames").Text = new LocString("credits", "LOC_THA.names").GetRawText();
		GetNode<MegaLabel>("%turHeader").Text = new LocString("credits", "LOC_TUR.header").GetRawText();
		tuple = SplitTwoColumn(new LocString("credits", "LOC_TUR.names").GetRawText());
		GetNode<MegaRichTextLabel>("%turRoles").Text = tuple.Item1;
		GetNode<MegaRichTextLabel>("%turNames").Text = tuple.Item2;
	}

	private void InitTwitchExtension()
	{
		GetNode<MegaLabel>("%TwitchHeader").Text = new LocString("credits", "TWITCH.header").GetRawText();
		(string, string) tuple = SplitTwoColumn(new LocString("credits", "TWITCH.names").GetRawText());
		GetNode<MegaRichTextLabel>("%TwitchRoles").Text = tuple.Item1;
		GetNode<MegaRichTextLabel>("%TwitchNames").Text = tuple.Item2;
	}

	private void InitModdingSupport()
	{
		GetNode<MegaLabel>("%ModdingSupportHeader").Text = new LocString("credits", "MODDING_SUPPORT.header").GetRawText();
		GetNode<MegaRichTextLabel>("%ModdingSupportNames").Text = ShuffleOneColumn(new LocString("credits", "MODDING_SUPPORT.names").GetRawText());
	}

	private void InitPlaytesters()
	{
		GetNode<MegaLabel>("%PlaytestersHeader").Text = new LocString("credits", "PLAYTESTERS.header").GetRawText();
		(string, string, string) tuple = SplitThreeColumnPlaytesters(new LocString("credits", "PLAYTESTERS.names").GetRawText());
		GetNode<MegaRichTextLabel>("%PlaytesterNames1").Text = tuple.Item1;
		GetNode<MegaRichTextLabel>("%PlaytesterNames2").Text = tuple.Item2;
		GetNode<MegaRichTextLabel>("%PlaytesterNames3").Text = tuple.Item3;
	}

	private void InitTrailer()
	{
		GetNode<MegaLabel>("%TrailerHeader").Text = new LocString("credits", "TRAILER.header").GetRawText();
		GetNode<MegaRichTextLabel>("%TrailerAnimationTeam").Text = new LocString("credits", "TRAILER_ANIMATION.team").GetRawText();
		GetNode<MegaLabel>("%TrailerAnimationHeader").Text = new LocString("credits", "TRAILER_ANIMATION.header").GetRawText();
		(string, string) tuple = SplitTwoColumn(new LocString("credits", "TRAILER_ANIMATION.names").GetRawText());
		GetNode<MegaRichTextLabel>("%TrailerAnimationRoles").Text = tuple.Item1;
		GetNode<MegaRichTextLabel>("%TrailerAnimationNames").Text = tuple.Item2;
		GetNode<MegaLabel>("%TrailerEditorHeader").Text = new LocString("credits", "TRAILER_EDITOR.header").GetRawText();
		tuple = SplitTwoColumn(new LocString("credits", "TRAILER_EDITOR.names").GetRawText());
		GetNode<MegaRichTextLabel>("%TrailerEditorRoles").Text = tuple.Item1;
		GetNode<MegaRichTextLabel>("%TrailerEditorNames").Text = tuple.Item2;
	}

	private void InitFmod()
	{
		GetNode<MegaLabel>("%FmodHeader").Text = new LocString("credits", "FMOD").GetRawText();
	}

	private void InitSpine()
	{
		GetNode<MegaLabel>("%SpineHeader").Text = new LocString("credits", "SPINE").GetRawText();
	}

	private void InitGodot()
	{
		GetNode<MegaLabel>("%GodotHeader").Text = new LocString("credits", "GODOT").GetRawText();
	}

	private void InitExitMessage()
	{
		GetNode<MegaRichTextLabel>("%ExitMessage").Text = new LocString("credits", "EXIT_MESSAGE").GetRawText();
	}

	public override void _EnterTree()
	{
		NHotkeyManager.Instance.AddBlockingScreen(this);
		NHotkeyManager.Instance.PushHotkeyReleasedBinding(MegaInput.cancel, CloseScreenDebug);
	}

	public override void _ExitTree()
	{
		NHotkeyManager.Instance.RemoveHotkeyReleasedBinding(MegaInput.cancel, CloseScreenDebug);
		NHotkeyManager.Instance.RemoveBlockingScreen(this);
	}

	private async Task EnableScreenExit()
	{
		await Task.Delay(2000);
		_canClose = true;
	}

	private void CloseScreenDebug()
	{
		if (_canClose && !_exitingScreen)
		{
			_exitingScreen = true;
			TaskHelper.RunSafely(FadeAndExitScreen());
		}
	}

	private async Task FadeAndExitScreen()
	{
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_screenContents, "modulate:a", 0f, 1.0);
		await ToSignal(_tween, Tween.SignalName.Finished);
		NModalContainer.Instance.Clear();
	}

	public override void _GuiInput(InputEvent inputEvent)
	{
		if (inputEvent is InputEventMouseButton inputEventMouseButton && inputEventMouseButton.ButtonIndex == MouseButton.Left && inputEventMouseButton.Pressed)
		{
			CloseScreenDebug();
		}
		ProcessScrollEvent(inputEvent);
	}

	public override void _Process(double delta)
	{
		float num = (float)delta;
		_targetPosition -= 80f * num;
		_screenContents.Position = _screenContents.Position.Lerp(new Vector2(_screenContents.Position.X, _targetPosition), num * 20f);
	}

	private void ProcessScrollEvent(InputEvent inputEvent)
	{
		if (inputEvent is InputEventMouseButton inputEventMouseButton)
		{
			if (inputEventMouseButton.ButtonIndex == MouseButton.WheelUp)
			{
				_targetPosition += 50f;
			}
			else if (inputEventMouseButton.ButtonIndex == MouseButton.WheelDown)
			{
				_targetPosition -= 50f;
			}
		}
		else if (inputEvent is InputEventPanGesture inputEventPanGesture)
		{
			_targetPosition += (0f - inputEventPanGesture.Delta.Y) * 20f;
		}
	}

	private static string ShuffleOneColumn(string input)
	{
		if (string.IsNullOrWhiteSpace(input))
		{
			return string.Empty;
		}
		List<string> list = input.Split(new string[1] { "||" }, StringSplitOptions.None).ToList();
		for (int num = list.Count - 1; num > 0; num--)
		{
			int num2 = Rng.Chaotic.NextInt(num + 1);
			List<string> list2 = list;
			int index = num;
			int index2 = num2;
			string value = list[num2];
			string value2 = list[num];
			list2[index] = value;
			list[index2] = value2;
		}
		return string.Join("\n", list);
	}

	private (string Roles, string Names) SplitTwoColumn(string input)
	{
		List<string> list = new List<string>();
		List<string> list2 = new List<string>();
		string[] array = input.Split(new char[1] { '\n' }, StringSplitOptions.RemoveEmptyEntries);
		string[] array2 = array;
		foreach (string text in array2)
		{
			string[] array3 = (from p in text.Split(new string[1] { "||" }, StringSplitOptions.RemoveEmptyEntries)
				select p.Trim() into p
				where !string.IsNullOrWhiteSpace(p)
				select p).ToArray();
			if (array3.Length == 2)
			{
				list.Add(array3[0]);
				list2.Add(array3[1]);
			}
		}
		return (Roles: string.Join("\n", list), Names: string.Join("\n", list2));
	}

	private (string Column1, string Column2, string Column3) SplitThreeColumnPlaytesters(string input)
	{
		string[] array = (from p in input.Split(new string[1] { "||" }, StringSplitOptions.RemoveEmptyEntries)
			select p.Trim() into p
			where !string.IsNullOrWhiteSpace(p)
			select p).ToArray();
		for (int num = array.Length - 1; num > 0; num--)
		{
			int num2 = Rng.Chaotic.NextInt(num + 1);
			ref string reference = ref array[num];
			ref string reference2 = ref array[num2];
			string text = array[num2];
			string text2 = array[num];
			reference = text;
			reference2 = text2;
		}
		List<string> list = new List<string>();
		List<string> list2 = new List<string>();
		List<string> list3 = new List<string>();
		for (int num3 = 0; num3 < array.Length; num3++)
		{
			switch (num3 % 3)
			{
			case 0:
				list.Add(array[num3]);
				break;
			case 1:
				list2.Add(array[num3]);
				break;
			case 2:
				list3.Add(array[num3]);
				break;
			}
		}
		return (Column1: string.Join("\n", list), Column2: string.Join("\n", list2), Column3: string.Join("\n", list3));
	}

	private static (string left, string right) SplitTwoColumnMultiRole(string input)
	{
		List<string> list = new List<string>();
		List<string> list2 = new List<string>();
		string[] array = input.Split(new char[1] { '\n' }, StringSplitOptions.RemoveEmptyEntries);
		for (int i = 0; i < array.Length; i++)
		{
			string[] array2 = array[i].Split(new string[1] { "||" }, StringSplitOptions.None);
			if (array2.Length == 2)
			{
				string text = array2[0].Trim();
				string text2 = array2[1].Trim();
				List<string> list3 = (from r in text.Split(new char[1] { ',' }, StringSplitOptions.RemoveEmptyEntries)
					select r.Trim()).ToList();
				for (int num = 0; num < list3.Count; num++)
				{
					list.Add(list3[num]);
					list2.Add((num == 0) ? text2 : "");
				}
				bool flag = list3.Count > 1;
				bool flag2 = i == array.Length - 1;
				if (flag && !flag2)
				{
					list.Add("");
					list2.Add("");
				}
			}
		}
		return (left: string.Join("\n", list), right: string.Join("\n", list2));
	}
}
