using Godot;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.CommonUi;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NLanguageDropdownItem : NDropdownItem
{
	public const string languageWarningIconPath = "res://images/ui/language_warning.png";

	private const string _warnImageTag = "[img]res://images/ui/language_warning.png[/img]";

	public string LanguageCode { get; private set; }

	public void Init(string languageCode)
	{
		LanguageCode = languageCode;
		string text = NLanguageDropdown.GetLanguageNameForCode(languageCode);
		float languageCompletion = LocManager.Instance.GetLanguageCompletion(languageCode);
		if (languageCompletion < 0.9f)
		{
			text = "[img]res://images/ui/language_warning.png[/img]" + text;
		}
		_richLabel.SetTextAutoSize(text);
	}
}
