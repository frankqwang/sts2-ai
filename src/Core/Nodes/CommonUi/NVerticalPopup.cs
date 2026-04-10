using System;
using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.CommonUi;

public partial class NVerticalPopup : Control
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("ui/vertical_popup");

	private Callable? _yesCallable;

	private Callable? _noCallable;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public MegaLabel TitleLabel { get; private set; }

	public MegaRichTextLabel BodyLabel { get; private set; }

	public NPopupYesNoButton YesButton { get; private set; }

	public NPopupYesNoButton NoButton { get; private set; }

	public override void _Ready()
	{
		TitleLabel = GetNode<MegaLabel>("Header");
		BodyLabel = GetNode<MegaRichTextLabel>("Description");
		YesButton = GetNode<NPopupYesNoButton>("YesButton");
		NoButton = GetNode<NPopupYesNoButton>("NoButton");
	}

	public void SetText(LocString title, LocString body)
	{
		TitleLabel.SetTextAutoSize(title.GetFormattedText());
		BodyLabel.Text = "[center]" + body.GetFormattedText() + "[/center]";
	}

	public void SetText(string title, string body)
	{
		TitleLabel.SetTextAutoSize(title);
		BodyLabel.Text = "[center]" + body + "[/center]";
	}

	public void InitYesButton(LocString yesButton, Action<NButton> onPressed)
	{
		_yesCallable = Callable.From(onPressed);
		YesButton.IsYes = true;
		YesButton.SetText(yesButton.GetFormattedText());
		YesButton.Connect(NClickableControl.SignalName.Released, _yesCallable.Value);
		YesButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(Close));
	}

	public void InitNoButton(LocString noButton, Action<NButton> onPressed)
	{
		_noCallable = Callable.From(onPressed);
		NoButton.Visible = true;
		NoButton.IsYes = false;
		NoButton.SetText(noButton.GetFormattedText());
		NoButton.Connect(NClickableControl.SignalName.Released, _noCallable.Value);
		NoButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(Close));
	}

	private void Close(NButton _)
	{
		NModalContainer.Instance.Clear();
	}

	public void HideNoButton()
	{
		NoButton.Visible = false;
	}

	public void DisconnectSignals()
	{
		if (_yesCallable.HasValue)
		{
			YesButton.Disconnect(NClickableControl.SignalName.Released, _yesCallable.Value);
			YesButton.Disconnect(NClickableControl.SignalName.Released, Callable.From<NButton>(Close));
		}
		if (_noCallable.HasValue)
		{
			NoButton.Disconnect(NClickableControl.SignalName.Released, _noCallable.Value);
			NoButton.Disconnect(NClickableControl.SignalName.Released, Callable.From<NButton>(Close));
		}
	}

	public void DisconnectHotkeys()
	{
		if (_yesCallable.HasValue)
		{
			YesButton.DisconnectHotkeys();
		}
		if (_noCallable.HasValue)
		{
			NoButton.DisconnectHotkeys();
		}
	}
}
