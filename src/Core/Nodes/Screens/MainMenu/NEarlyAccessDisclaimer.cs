using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;

public partial class NEarlyAccessDisclaimer : Control, IScreenContext
{
	private Tween? _tween;

	private Control _image;

	public Control? DefaultFocusedControl => null;

	public static NEarlyAccessDisclaimer Create()
	{
		NEarlyAccessDisclaimer nEarlyAccessDisclaimer = ResourceLoader.Load<PackedScene>("res://scenes/screens/main_menu/early_access_disclaimer.tscn", null, ResourceLoader.CacheMode.Reuse).Instantiate<NEarlyAccessDisclaimer>(PackedScene.GenEditState.Disabled);
		NHotkeyManager.Instance?.AddBlockingScreen(nEarlyAccessDisclaimer);
		return nEarlyAccessDisclaimer;
	}

	public override void _Ready()
	{
		string formattedText = new LocString("main_menu_ui", "EARLY_ACCESS_DISCLAIMER.header").GetFormattedText();
		GetNode<MegaLabel>("%Header").SetTextAutoSize(formattedText);
		_image = GetNode<TextureRect>("Image");
		UpdateEaDisclaimerDescription();
		NControllerManager.Instance.Connect(NControllerManager.SignalName.MouseDetected, Callable.From(UpdateEaDisclaimerDescription));
		NControllerManager.Instance.Connect(NControllerManager.SignalName.ControllerDetected, Callable.From(UpdateEaDisclaimerDescription));
	}

	public async Task CloseScreen()
	{
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_image, "modulate:a", 1f, 0.5);
		_tween.TweenProperty(_image, "position:y", _image.Position.Y - 1000f, 0.5).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Back);
		await ToSignal(_tween, Tween.SignalName.Finished);
		SaveManager.Instance.SettingsSave.SeenEaDisclaimer = true;
		this.QueueFreeSafely();
		NHotkeyManager.Instance?.RemoveBlockingScreen(this);
		NModalContainer.Instance.Clear();
	}

	private void UpdateEaDisclaimerDescription()
	{
		string textAutoSize = ((!NControllerManager.Instance.IsUsingController) ? new LocString("main_menu_ui", "EARLY_ACCESS_DISCLAIMER.description_mkb").GetFormattedText() : new LocString("main_menu_ui", "EARLY_ACCESS_DISCLAIMER.description_controller").GetFormattedText());
		GetNode<MegaRichTextLabel>("%Description").SetTextAutoSize(textAutoSize);
	}
}
