using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Settings;

public partial class NAbandonRunButton : NButton
{
	private MegaLabel _label;

	public override void _Ready()
	{
		ConnectSignals();
		_label = GetNode<MegaLabel>("Label");
	}

	protected override void OnRelease()
	{
		NModalContainer.Instance.Add(NAbandonRunConfirmPopup.Create(null));
	}

	protected override void OnFocus()
	{
		_label.Modulate = StsColors.gold;
	}

	protected override void OnUnfocus()
	{
		_label.Modulate = StsColors.cream;
	}
}
