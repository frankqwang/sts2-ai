using System;
using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Vfx;
using MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;
using MegaCrit.Sts2.Core.Random;

namespace MegaCrit.Sts2.Core.Nodes.Rooms;

public partial class NMerchantButton : NButton
{
	[Signal]
	public delegate void MerchantOpenedEventHandler(NMerchantButton merchantButton);

	private MegaSkeleton _merchantSkeleton;

	private NSelectionReticle _merchantSelectionReticle;

	private bool _focusedWhileTargeting;

	protected override string[] Hotkeys => new string[1] { MegaInput.select };

	public bool IsLocalPlayerDead { get; set; }

	public IReadOnlyList<LocString> PlayerDeadLines { get; set; } = Array.Empty<LocString>();

	public override void _Ready()
	{
		ConnectSignals();
		_merchantSelectionReticle = GetNode<NSelectionReticle>("%MerchantSelectionReticle");
		MegaSprite megaSprite = new MegaSprite(GetNode("%MerchantVisual"));
		_merchantSkeleton = megaSprite.GetSkeleton();
		megaSprite.GetAnimationState().SetAnimation("idle_loop");
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		RefreshFocus();
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		_merchantSelectionReticle.OnDeselect();
		if (_focusedWhileTargeting)
		{
			NTargetManager.Instance.OnNodeUnhovered(this);
		}
		else
		{
			_merchantSkeleton.SetSkinByName("default");
			_merchantSkeleton.SetSlotsToSetupPose();
		}
		_focusedWhileTargeting = false;
	}

	protected override void OnRelease()
	{
		if (_focusedWhileTargeting)
		{
			_merchantSelectionReticle.OnDeselect();
			_focusedWhileTargeting = false;
			RefreshFocus();
		}
		else if (IsLocalPlayerDead)
		{
			LocString locString = Rng.Chaotic.NextItem(PlayerDeadLines);
			if (locString != null)
			{
				PlayDialogue(locString);
			}
		}
		else
		{
			EmitSignalMerchantOpened(this);
		}
	}

	public NSpeechBubbleVfx? PlayDialogue(LocString line, double duration = 2.0)
	{
		NSpeechBubbleVfx nSpeechBubbleVfx = NSpeechBubbleVfx.Create(line.GetFormattedText(), DialogueSide.Right, base.GlobalPosition + base.Size.X * Vector2.Left, duration, VfxColor.Blue);
		if (nSpeechBubbleVfx != null)
		{
			GetParent().AddChildSafely(nSpeechBubbleVfx);
		}
		return nSpeechBubbleVfx;
	}

	private void RefreshFocus()
	{
		if (NTargetManager.Instance.IsInSelection && NTargetManager.Instance.AllowedToTargetNode(this))
		{
			NTargetManager.Instance.OnNodeHovered(this);
			_merchantSelectionReticle.OnSelect();
			_focusedWhileTargeting = true;
		}
		else
		{
			_merchantSkeleton.SetSkinByName("outline");
			_merchantSkeleton.SetSlotsToSetupPose();
			_focusedWhileTargeting = false;
		}
	}
}
