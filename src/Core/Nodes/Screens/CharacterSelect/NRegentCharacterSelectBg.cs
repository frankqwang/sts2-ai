using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;

namespace MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;

public partial class NRegentCharacterSelectBg : Control
{
	private MegaSprite _spineController;

	private Control _sphereGuardianHover;

	private Control _decaHover;

	private Control _sentryHover;

	private Control _sneckoHover;

	private Control _cultistHover;

	private Control _shapesHover;

	private Control _amongusHover;

	public override void _Ready()
	{
		_spineController = new MegaSprite(GetNode("SpineSprite"));
		_sphereGuardianHover = GetNode<Control>("SphereGuardianHover");
		_sphereGuardianHover.Connect(Control.SignalName.MouseEntered, Callable.From(delegate
		{
			SetSkin("spheric guardian constellation");
		}));
		_sphereGuardianHover.Connect(Control.SignalName.MouseExited, Callable.From(delegate
		{
			SetSkin("normal");
		}));
		_decaHover = GetNode<Control>("DecaHover");
		_decaHover.Connect(Control.SignalName.MouseEntered, Callable.From(delegate
		{
			SetSkin("deca outline");
		}));
		_decaHover.Connect(Control.SignalName.MouseExited, Callable.From(delegate
		{
			SetSkin("normal");
		}));
		_sentryHover = GetNode<Control>("SentryHover");
		_sentryHover.Connect(Control.SignalName.MouseEntered, Callable.From(delegate
		{
			SetSkin("sentry constellation");
		}));
		_sentryHover.Connect(Control.SignalName.MouseExited, Callable.From(delegate
		{
			SetSkin("normal");
		}));
		_sneckoHover = GetNode<Control>("SneckoHover");
		_sneckoHover.Connect(Control.SignalName.MouseEntered, Callable.From(delegate
		{
			SetSkin("snecko constellation");
		}));
		_sneckoHover.Connect(Control.SignalName.MouseExited, Callable.From(delegate
		{
			SetSkin("normal");
		}));
		_cultistHover = GetNode<Control>("CultistHover");
		_cultistHover.Connect(Control.SignalName.MouseEntered, Callable.From(delegate
		{
			SetSkin("cultist constellation");
		}));
		_cultistHover.Connect(Control.SignalName.MouseExited, Callable.From(delegate
		{
			SetSkin("normal");
		}));
		_shapesHover = GetNode<Control>("ShapesHover");
		_shapesHover.Connect(Control.SignalName.MouseEntered, Callable.From(delegate
		{
			SetSkin("shapes constellation");
		}));
		_shapesHover.Connect(Control.SignalName.MouseExited, Callable.From(delegate
		{
			SetSkin("normal");
		}));
		_amongusHover = GetNode<Control>("AmongusHover");
		_amongusHover.Connect(Control.SignalName.MouseEntered, Callable.From(delegate
		{
			SetSkin("amongus constellation");
		}));
		_amongusHover.Connect(Control.SignalName.MouseExited, Callable.From(delegate
		{
			SetSkin("normal");
		}));
	}

	private void SetSkin(string skinName)
	{
		MegaSkeleton skeleton = _spineController.GetSkeleton();
		skeleton.SetSkin(skeleton.GetData().FindSkin(skinName));
		skeleton.SetSlotsToSetupPose();
	}
}
