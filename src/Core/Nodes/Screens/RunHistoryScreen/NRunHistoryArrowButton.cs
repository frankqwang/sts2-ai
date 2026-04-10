using Godot;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Nodes.CommonUi;

namespace MegaCrit.Sts2.Core.Nodes.Screens.RunHistoryScreen;

public partial class NRunHistoryArrowButton : NGoldArrowButton
{
	private bool _isLeft;

	public bool IsLeft
	{
		get
		{
			return _isLeft;
		}
		set
		{
			if (_isLeft != value)
			{
				UnregisterHotkeys();
				_isLeft = value;
				RegisterHotkeys();
				_icon.FlipH = !IsLeft;
				UpdateControllerButton();
			}
		}
	}

	protected override string[] Hotkeys => new string[1] { _isLeft ? MegaInput.viewDeckAndTabLeft : MegaInput.viewExhaustPileAndTabRight };

	public override void _Ready()
	{
		base._Ready();
		_icon.FlipH = !IsLeft;
	}
}
