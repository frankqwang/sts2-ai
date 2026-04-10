using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Multiplayer;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.ProfileScreen;

public partial class NDeleteProfileButton : NButton
{
	private static readonly StringName _v = new StringName("v");

	private static readonly LocString _title = new LocString("main_menu_ui", "PROFILE_SCREEN.DELETE_CONFIRM_POPUP.title");

	private static readonly LocString _description = new LocString("main_menu_ui", "PROFILE_SCREEN.DELETE_CONFIRM_POPUP.description");

	private static readonly LocString _buttonMesssage = new LocString("main_menu_ui", "PROFILE_SCREEN.DELETE_BUTTON.label");

	private TextureRect _icon;

	private MegaLabel _label;

	private ShaderMaterial _hsv;

	private Tween? _tween;

	private NProfileScreen _profileScreen;

	private int _profileId;

	public override void _Ready()
	{
		ConnectSignals();
		_label = GetNode<MegaLabel>("%MegaLabel");
		_label.SetTextAutoSize(_buttonMesssage.GetFormattedText());
		_icon = GetNode<TextureRect>("Icon");
		_hsv = (ShaderMaterial)_icon.Material;
	}

	public void Initialize(NProfileScreen profileScreen, int profileId)
	{
		_profileScreen = profileScreen;
		_profileId = profileId;
		bool visible = NProfileScreen.forceShowProfileAsDeleted != profileId && FileAccess.FileExists(UserDataPathProvider.GetProfileScopedPath(profileId, "saves/progress.save"));
		base.Visible = visible;
	}

	protected override void OnRelease()
	{
		TaskHelper.RunSafely(ConfirmDeletion());
	}

	private async Task ConfirmDeletion()
	{
		NGenericPopup nGenericPopup = NGenericPopup.Create();
		NModalContainer.Instance.Add(nGenericPopup);
		_title.Add("Id", _profileId);
		_description.Add("Id", _profileId);
		if (await nGenericPopup.WaitForConfirmation(_description, _title, new LocString("main_menu_ui", "PROFILE_SCREEN.DELETE_CONFIRM_POPUP.cancel"), new LocString("main_menu_ui", "PROFILE_SCREEN.DELETE_CONFIRM_POPUP.delete")))
		{
			Log.Info($"Player clicked yes on confirm deletion popup for {_profileId}");
			SaveManager.Instance.DeleteProfile(_profileId);
			NProfileScreen.forceShowProfileAsDeleted = _profileId;
			SaveManager.Instance.InitProgressData();
			SaveManager.Instance.InitPrefsData();
			if (_profileId == SaveManager.Instance.CurrentProfileId)
			{
				NGame.Instance.ReloadMainMenu();
				Callable.From(NGame.Instance.MainMenu.OpenProfileScreen).CallDeferred();
			}
			else
			{
				_profileScreen.Refresh();
			}
		}
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "scale", Vector2.One * 1.1f, 0.05);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 1.4f, 0.05);
		_tween.TweenProperty(_label, "modulate:a", 1f, 0.2);
		_tween.TweenProperty(_label, "position:y", 78f, 0.2).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic)
			.From(48f);
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_tween.TweenProperty(_label, "modulate:a", 0f, 0.05);
	}

	private void UpdateShaderV(float value)
	{
		_hsv.SetShaderParameter(_v, value);
	}
}
