using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Debug;
using MegaCrit.Sts2.Core.DevConsole.ConsoleCommands;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Platform;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;
using Sentry;

namespace MegaCrit.Sts2.Core.Nodes.Screens.FeedbackScreen;

public partial class NSendFeedbackScreen : Control, IScreenContext
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/feedback_screen/feedback_screen");

	private const float _superWiggleTime = 0.25f;

	private const string _defaultUrl = "https://feedback.sts2.megacrit.com/feedback";

	private static readonly string _url = System.Environment.GetEnvironmentVariable("STS2_FEEDBACK_URL") ?? "https://feedback.sts2.megacrit.com/feedback";

	private static readonly System.Net.Http.HttpClient _httpClient = new System.Net.Http.HttpClient();

	private const int _maxDescriptionChars = 500;

	private NBackButton _backButton;

	private Control _mainPanel;

	private NMegaTextEdit _descriptionInput;

	private MegaLabel _emojiLabel;

	private NButton _sendButton;

	private MegaLabel _sendLabel;

	private NFeedbackCategoryDropdown _categoryDropdown;

	private Control _successBackstop;

	private Control _successPanel;

	private MegaLabel _successLabel;

	private List<NSendFeedbackCartoon> _cartoons = new List<NSendFeedbackCartoon>();

	private NSendFeedbackFlower _flower;

	private CancellationTokenSource? _cancelToken;

	private CancellationTokenSource? _sendCancelToken;

	private readonly List<NSendFeedbackEmojiButton> _emojiButtons = new List<NSendFeedbackEmojiButton>();

	private NSendFeedbackEmojiButton? _selectedEmoteButton;

	private byte[]? _screenshotBytes;

	private Vector2 _originalSuccessPosition;

	private ulong _lastClosedMsec;

	private string _descriptionText = string.Empty;

	private int _descriptionCaretLine;

	private int _descriptionCaretColumn;

	private Tween? _wiggleTween;

	public Control DefaultFocusedControl => _descriptionInput;

	public static NSendFeedbackScreen? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NSendFeedbackScreen>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		_mainPanel = GetNode<Control>("%MainPanel");
		_descriptionInput = GetNode<NMegaTextEdit>("%DescriptionInput");
		_emojiLabel = GetNode<MegaLabel>("%EmojiLabel");
		_sendButton = GetNode<NButton>("%SendButton");
		_sendLabel = _sendButton.GetNode<MegaLabel>("Label");
		_categoryDropdown = GetNode<NFeedbackCategoryDropdown>("%CategoryDropdown");
		_successBackstop = GetNode<Control>("%SuccessBackstop");
		_successPanel = GetNode<Control>("%SuccessPanel");
		_successLabel = GetNode<MegaLabel>("%SuccessLabel");
		_backButton = GetNode<NBackButton>("BackButton");
		_originalSuccessPosition = _successPanel.Position;
		int num = 3;
		List<NSendFeedbackCartoon> list = new List<NSendFeedbackCartoon>(num);
		CollectionsMarshal.SetCount(list, num);
		Span<NSendFeedbackCartoon> span = CollectionsMarshal.AsSpan(list);
		int num2 = 0;
		span[num2] = GetNode<NSendFeedbackCartoon>("Sun");
		num2++;
		span[num2] = GetNode<NSendFeedbackCartoon>("Cupcake");
		num2++;
		span[num2] = GetNode<NSendFeedbackCartoon>("FlowerContainer/Flower");
		_cartoons = list;
		_flower = GetNode<NSendFeedbackFlower>("FlowerContainer");
		foreach (Node child in GetNode("%EmojiButtonContainer").GetChildren())
		{
			if (child is NSendFeedbackEmojiButton nSendFeedbackEmojiButton)
			{
				_emojiButtons.Add(nSendFeedbackEmojiButton);
				nSendFeedbackEmojiButton.PivotOffset = nSendFeedbackEmojiButton.Size * 0.5f;
				nSendFeedbackEmojiButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(EmojiButtonSelected));
			}
		}
		_sendButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(SendButtonSelected));
		_sendButton.Connect(NClickableControl.SignalName.Focused, Callable.From<NClickableControl>(SendButtonFocused));
		_sendButton.Connect(NClickableControl.SignalName.Unfocused, Callable.From<NClickableControl>(SendButtonUnfocused));
		_descriptionInput.Connect(TextEdit.SignalName.TextChanged, Callable.From(OnDescriptionChanged));
		_sendButton.FocusNeighborTop = _categoryDropdown.GetPath();
		_sendButton.FocusNeighborLeft = _emojiButtons.Last().GetPath();
		_sendButton.FocusNeighborBottom = _sendButton.GetPath();
		_sendButton.FocusNeighborRight = _sendButton.GetPath();
		_emojiButtons.Last().FocusNeighborRight = _sendButton.GetPath();
		foreach (NSendFeedbackEmojiButton emojiButton in _emojiButtons)
		{
			emojiButton.FocusNeighborTop = _categoryDropdown.GetPath();
			emojiButton.FocusNeighborBottom = emojiButton.GetPath();
		}
		_categoryDropdown.FocusNeighborRight = _sendButton.GetPath();
		_categoryDropdown.FocusNeighborBottom = _emojiButtons.First().GetPath();
		_categoryDropdown.FocusNeighborTop = _descriptionInput.GetPath();
		_descriptionInput.FocusNeighborTop = _descriptionInput.GetPath();
		_descriptionInput.FocusNeighborLeft = _descriptionInput.GetPath();
		_descriptionInput.FocusNeighborRight = _descriptionInput.GetPath();
		_descriptionInput.FocusNeighborBottom = _categoryDropdown.GetPath();
		_backButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(delegate
		{
			Close();
		}));
		base.Visible = false;
		base.MouseFilter = MouseFilterEnum.Ignore;
		_backButton.Disable();
	}

	public void Relocalize()
	{
		_descriptionInput.PlaceholderText = new LocString("settings_ui", "FEEDBACK_DESCRIPTION_PLACEHOLDER").GetFormattedText();
		_emojiLabel.SetTextAutoSize(new LocString("settings_ui", "FEEDBACK_EMOJI_LABEL").GetFormattedText());
		_sendLabel.SetTextAutoSize(new LocString("settings_ui", "FEEDBACK_SEND_BUTTON_LABEL").GetFormattedText());
		_descriptionInput.RefreshFont();
		_emojiLabel.RefreshFont();
		_sendLabel.RefreshFont();
	}

	private void OnDescriptionChanged()
	{
		if (_descriptionInput.Text.Length > 500)
		{
			_descriptionInput.Text = _descriptionText;
			_descriptionInput.SetCaretLine(_descriptionCaretLine);
			_descriptionInput.SetCaretColumn(_descriptionCaretColumn);
		}
		else
		{
			_descriptionText = _descriptionInput.Text;
			_descriptionCaretLine = _descriptionInput.GetCaretLine();
			_descriptionCaretColumn = _descriptionInput.GetCaretColumn();
		}
	}

	public void SetScreenshot(Image screenshot)
	{
		int width = screenshot.GetWidth();
		int height = screenshot.GetHeight();
		float num = (float)width / (float)height;
		if (width > 1280)
		{
			screenshot.Resize(1280, Mathf.RoundToInt(1280f / num), Image.Interpolation.Bilinear);
		}
		if (height > 720)
		{
			screenshot.Resize(Mathf.RoundToInt(720f * num), 720, Image.Interpolation.Bilinear);
		}
		_screenshotBytes = screenshot.SavePngToBuffer();
	}

	private void EmojiButtonSelected(NButton button)
	{
		SetSelectedEmoji((NSendFeedbackEmojiButton)button);
	}

	private void SendButtonFocused(NClickableControl _)
	{
		_flower.SetState(NSendFeedbackFlower.State.Anticipation);
	}

	private void SendButtonUnfocused(NClickableControl _)
	{
		if (_flower.MyState == NSendFeedbackFlower.State.Anticipation)
		{
			_flower.SetState(NSendFeedbackFlower.State.None);
		}
	}

	public void Open()
	{
		Log.Info("Feedback screen opened");
		if (Time.GetTicksMsec() - _lastClosedMsec > 60000)
		{
			ClearInput();
		}
		base.Visible = true;
		_flower.SetState(NSendFeedbackFlower.State.None);
		_successBackstop.Visible = false;
		base.MouseFilter = MouseFilterEnum.Stop;
		NHotkeyManager.Instance.AddBlockingScreen(this);
		ActiveScreenContext.Instance.Update();
		_backButton.Enable();
	}

	private void Close()
	{
		Log.Info("Feedback screen closed");
		_flower.SetState(NSendFeedbackFlower.State.None);
		_successBackstop.Visible = false;
		_mainPanel.Modulate = Colors.White;
		_wiggleTween?.Kill();
		base.Visible = false;
		_lastClosedMsec = Time.GetTicksMsec();
		_cancelToken?.Cancel();
		base.MouseFilter = MouseFilterEnum.Ignore;
		_backButton.Disable();
		NHotkeyManager.Instance.RemoveBlockingScreen(this);
		ActiveScreenContext.Instance.Update();
	}

	private void ClearInput()
	{
		_descriptionInput.Text = string.Empty;
		_descriptionText = string.Empty;
		SetSelectedEmoji(null);
	}

	private void SetSelectedEmoji(NSendFeedbackEmojiButton? button)
	{
		NSendFeedbackEmojiButton selectedEmoteButton = _selectedEmoteButton;
		_selectedEmoteButton?.SetSelected(isSelected: false);
		if (selectedEmoteButton != button)
		{
			_selectedEmoteButton = button;
			_selectedEmoteButton?.SetSelected(isSelected: true);
		}
	}

	private void SendButtonSelected(NButton _)
	{
		Log.Info("Beginning asynchronous feedback send at " + Log.Timestamp + ": " + _descriptionText);
		ReleaseInfo releaseInfo = ReleaseInfoManager.Instance.ReleaseInfo;
		string text = releaseInfo?.Commit ?? GitHelper.ShortCommitId;
		FeedbackData data = new FeedbackData
		{
			description = _descriptionText,
			category = _categoryDropdown.CurrentCategory,
			gameVersion = (releaseInfo?.Version ?? GitHelper.ShortCommitId ?? "unknown"),
			uniqueId = SaveManager.Instance.Progress.UniqueId,
			commit = (text ?? "unknown"),
			platformBranch = PlatformUtil.GetPlatformBranch(),
			sessionId = SentryService.SessionId
		};
		byte[] screenshotBytes = _screenshotBytes;
		int currentProfileId = SaveManager.Instance.CurrentProfileId;
		_sendCancelToken?.Cancel();
		_sendCancelToken?.Dispose();
		_sendCancelToken = new CancellationTokenSource();
		TaskHelper.RunSafely(SendFeedback(data, screenshotBytes, currentProfileId, _sendCancelToken.Token));
		ClearInput();
		_screenshotBytes = null;
		TaskHelper.RunSafely(OnFeedbackSuccess());
	}

	private static async Task SendFeedback(FeedbackData data, byte[] screenshotBytes, int profileId, CancellationToken cancellationToken)
	{
		if (string.IsNullOrEmpty(data.description))
		{
			return;
		}
		using MemoryStream logsMemoryStream = new MemoryStream();
		GetLogsConsoleCmd.ZipFeedbackLogs(logsMemoryStream, profileId);
		byte[] logsZipBytes = logsMemoryStream.ToArray();
		using MultipartFormDataContent formContent = BuildMultipartContent(data, screenshotBytes, logsZipBytes);
		byte[] bodyBytes = await formContent.ReadAsByteArrayAsync(cancellationToken);
		MediaTypeHeaderValue contentType = formContent.Headers.ContentType;
		int[] delaysMs = new int[3] { 1000, 2000, 4000 };
		string sentryMessage = null;
		for (int attempt = 0; attempt <= 3; attempt++)
		{
			cancellationToken.ThrowIfCancellationRequested();
			try
			{
				using ByteArrayContent content = new ByteArrayContent(bodyBytes);
				content.Headers.ContentType = contentType;
				using HttpResponseMessage response = await _httpClient.PutAsync(_url, content, cancellationToken);
				if (response.IsSuccessStatusCode)
				{
					Log.Info("Feedback successfully posted!");
					return;
				}
				int statusCode = (int)response.StatusCode;
				if (statusCode >= 400 && statusCode < 500 && statusCode != 429)
				{
					string value = await response.Content.ReadAsStringAsync(cancellationToken);
					Log.Warn($"Feedback rejected ({response.StatusCode}): {value}");
					SentrySdk.CaptureMessage($"Feedback rejected: Response status code {response.StatusCode}");
					return;
				}
				sentryMessage = $"Response status code {response.StatusCode}";
				Log.Warn($"Feedback attempt {attempt + 1}/{4} failed: {response.StatusCode}");
			}
			catch (HttpRequestException ex)
			{
				string text = $"Feedback attempt {attempt + 1}/{4} network error: {ExceptionMessageWithInner(ex)} {ex.HttpRequestError}";
				if (ex.HttpRequestError != HttpRequestError.NameResolutionError)
				{
					sentryMessage = "HttpRequestException: " + ExceptionMessageWithInner(ex);
				}
				Log.Warn(text);
			}
			if (attempt < 3)
			{
				await Task.Delay(delaysMs[attempt], cancellationToken);
			}
		}
		Log.Warn("Feedback send failed after all retry attempts");
		if (sentryMessage != null)
		{
			SentrySdk.CaptureMessage("Feedback failed to send: " + sentryMessage);
		}
	}

	private static string ExceptionMessageWithInner(Exception ex)
	{
		if (ex.InnerException == null)
		{
			return ex.Message;
		}
		return ex.Message + " | " + ExceptionMessageWithInner(ex.InnerException);
	}

	private static MultipartFormDataContent BuildMultipartContent(FeedbackData data, byte[] screenshotBytes, byte[] logsZipBytes)
	{
		string content = JsonSerializer.Serialize(data, JsonSerializationUtility.GetTypeInfo<FeedbackData>());
		MultipartFormDataContent multipartFormDataContent = new MultipartFormDataContent();
		StringContent stringContent = new StringContent(content);
		stringContent.Headers.ContentType = new MediaTypeHeaderValue("application/json");
		stringContent.Headers.ContentDisposition = new ContentDispositionHeaderValue("form-data")
		{
			Name = "payload_json"
		};
		multipartFormDataContent.Add(stringContent);
		ByteArrayContent byteArrayContent = new ByteArrayContent(screenshotBytes);
		byteArrayContent.Headers.ContentType = new MediaTypeHeaderValue("image/png");
		byteArrayContent.Headers.ContentDisposition = new ContentDispositionHeaderValue("form-data")
		{
			Name = "screenshot"
		};
		multipartFormDataContent.Add(byteArrayContent);
		ByteArrayContent byteArrayContent2 = new ByteArrayContent(logsZipBytes);
		byteArrayContent2.Headers.ContentType = new MediaTypeHeaderValue("application/zip");
		byteArrayContent2.Headers.ContentDisposition = new ContentDispositionHeaderValue("form-data")
		{
			Name = "logs"
		};
		multipartFormDataContent.Add(byteArrayContent2);
		return multipartFormDataContent;
	}

	private async Task OnFeedbackSuccess()
	{
		_successBackstop.Visible = true;
		_successPanel.Modulate = Colors.Transparent;
		Control successPanel = _successPanel;
		Vector2 position = _successPanel.Position;
		position.Y = _originalSuccessPosition.Y + 20f;
		successPanel.Position = position;
		_successLabel.SetTextAutoSize(new LocString("settings_ui", "FEEDBACK_SEND_SUCCESS_LABEL").GetFormattedText());
		_successLabel.Modulate = StsColors.green;
		Tween tween = GetTree().CreateTween().Parallel();
		tween.TweenProperty(_mainPanel, "modulate", new Color(0.1f, 0.1f, 0.1f), 0.15000000596046448);
		tween.TweenProperty(_successPanel, "modulate", Colors.White, 0.15000000596046448);
		tween.TweenProperty(_successPanel, "position:y", _originalSuccessPosition.Y, 0.15000000596046448).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Quad);
		tween.Chain().TweenProperty(_successLabel, "position:y", _successLabel.Position.Y - 10f, 0.10000000149011612).SetEase(Tween.EaseType.Out)
			.SetTrans(Tween.TransitionType.Quad);
		tween.Chain().TweenProperty(_successLabel, "position:y", _successLabel.Position.Y, 0.10000000149011612).SetEase(Tween.EaseType.In)
			.SetTrans(Tween.TransitionType.Quad);
		_wiggleTween?.Kill();
		_wiggleTween = CreateTween();
		_wiggleTween.TweenCallback(Callable.From(WiggleCartoons1));
		_wiggleTween.TweenInterval(0.25);
		_wiggleTween.TweenCallback(Callable.From(WiggleCartoons2));
		_wiggleTween.TweenInterval(0.25);
		_wiggleTween.SetLoops();
		string scenePath = SceneHelper.GetScenePath("vfx/vfx_dramatic_entrance_fullscreen");
		Node2D node2D = PreloadManager.Cache.GetScene(scenePath).Instantiate<Node2D>(PackedScene.GenEditState.Disabled);
		this.AddChildSafely(node2D);
		MoveChild(node2D, 1);
		node2D.GlobalPosition = NGame.Instance.GetViewportRect().Size * 0.5f;
		_flower.SetState(NSendFeedbackFlower.State.NoddingFast);
		_cancelToken = new CancellationTokenSource();
		await Task.Delay(2000, _cancelToken.Token);
		Close();
	}

	private void WiggleCartoons1()
	{
		foreach (NSendFeedbackCartoon cartoon in _cartoons)
		{
			if (_flower.MyState == NSendFeedbackFlower.State.None || cartoon != _flower.Cartoon)
			{
				cartoon.SetRotation1();
			}
		}
	}

	private void WiggleCartoons2()
	{
		foreach (NSendFeedbackCartoon cartoon in _cartoons)
		{
			if (_flower.MyState == NSendFeedbackFlower.State.None || cartoon != _flower.Cartoon)
			{
				cartoon.SetRotation2();
			}
		}
	}
}
