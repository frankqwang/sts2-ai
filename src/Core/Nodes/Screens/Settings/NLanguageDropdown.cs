using System;
using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NLanguageDropdown : NSettingsDropdown
{
	[Export(PropertyHint.None, "")]
	private PackedScene _dropdownItemScene;

	private static readonly Dictionary<string, string> _languageCodeToName = new Dictionary<string, string>
	{
		{ "ARA", "العربية" },
		{ "BEN", "ব\u09be\u0982ল\u09be" },
		{ "CZE", "Čeština" },
		{ "DEU", "Deutsch" },
		{ "DUT", "Nederlands" },
		{ "ENG", "English" },
		{ "ESP", "Español (Latinoamérica)" },
		{ "FIL", "Filipino" },
		{ "FIN", "Suomi" },
		{ "FRA", "Français" },
		{ "GRE", "Ελληνικά" },
		{ "HIN", "ह\u093fन\u094dद\u0940" },
		{ "IND", "Bahasa Indonesia" },
		{ "ITA", "Italiano" },
		{ "JPN", "日本語" },
		{ "KOR", "한국어" },
		{ "MAL", "Bahasa Melayu" },
		{ "NOR", "Norsk" },
		{ "POL", "Polski" },
		{ "POR", "Português" },
		{ "PTB", "Português Brasileiro" },
		{ "RUS", "Русский" },
		{ "SPA", "Español (Castellano)" },
		{ "SWE", "Svenska" },
		{ "THA", "ไทย" },
		{ "TUR", "Türkçe" },
		{ "UKR", "Українська" },
		{ "VIE", "Tiếng Việt" },
		{ "ZHS", "中文" },
		{ "ZHT", "繁體中文" }
	};

	private string CurrentLanguage => LocManager.Instance.Language;

	public override void _Ready()
	{
		ConnectSignals();
		PopulateOptions();
		_currentOptionLabel.SetTextAutoSize(GetLanguageNameForCode(CurrentLanguage));
	}

	private void PopulateOptions()
	{
		ClearDropdownItems();
		foreach (string language in LocManager.Languages)
		{
			NLanguageDropdownItem nLanguageDropdownItem = _dropdownItemScene.Instantiate<NLanguageDropdownItem>(PackedScene.GenEditState.Disabled);
			_dropdownItems.AddChildSafely(nLanguageDropdownItem);
			nLanguageDropdownItem.Connect(NDropdownItem.SignalName.Selected, Callable.From<NDropdownItem>(OnDropdownItemSelected));
			nLanguageDropdownItem.Init(language);
		}
		_dropdownItems.GetParent<NDropdownContainer>().RefreshLayout();
	}

	private void OnDropdownItemSelected(NDropdownItem nDropdownItem)
	{
		NLanguageDropdownItem nLanguageDropdownItem = (NLanguageDropdownItem)nDropdownItem;
		if (!(nLanguageDropdownItem.LanguageCode == CurrentLanguage))
		{
			CloseDropdown();
			_currentOptionLabel.SetTextAutoSize(GetLanguageNameForCode(nLanguageDropdownItem.LanguageCode));
			SaveManager.Instance.SettingsSave.Language = nLanguageDropdownItem.LanguageCode;
			LocManager.Instance.SetLanguage(nLanguageDropdownItem.LanguageCode);
			NGame.Instance.Relocalize();
			NGame.Instance.MainMenu.OpenSettingsMenu();
		}
	}

	public static string GetLanguageNameForCode(string languageCode)
	{
		if (!_languageCodeToName.TryGetValue(languageCode.ToUpperInvariant(), out string value))
		{
			throw new InvalidOperationException("Tried to get language name for code " + languageCode + " but it doesn't exist!");
		}
		return value;
	}
}
