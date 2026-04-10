using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.CardLibrary;

public partial class NCardLibraryStats : Control
{
	private MegaRichTextLabel _label;

	private static LocString Victories => new LocString("card_library", "VICTORIES");

	private static string ScenePath => SceneHelper.GetScenePath("screens/card_library/card_library_stats");

	public static NCardLibraryStats Create()
	{
		return PreloadManager.Cache.GetScene(ScenePath).Instantiate<NCardLibraryStats>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		_label = GetNode<MegaRichTextLabel>("%Label");
	}

	public void UpdateStats(CardModel card)
	{
		LocString victories = Victories;
		CardStats value;
		long num = ((!SaveManager.Instance.Progress.CardStats.TryGetValue(card.Id, out value)) ? 0 : value.TimesWon);
		victories.Add("Victories", num);
		_label.Text = victories.GetFormattedText();
	}
}
