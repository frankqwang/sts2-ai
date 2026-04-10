using System;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Ancients;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Events;

public partial class NAncientDialogueLine : NButton
{
	private const string _scenePath = "res://scenes/events/ancient_dialogue_line.tscn";

	private AncientDialogueLine _line;

	private AncientEventModel _ancient;

	private CharacterModel _character;

	private Control _iconNode;

	private Tween? _tween;

	private float _targetAlpha = 1f;

	public static NAncientDialogueLine Create(AncientDialogueLine line, AncientEventModel ancient, CharacterModel character)
	{
		NAncientDialogueLine nAncientDialogueLine = PreloadManager.Cache.GetScene("res://scenes/events/ancient_dialogue_line.tscn").Instantiate<NAncientDialogueLine>(PackedScene.GenEditState.Disabled);
		nAncientDialogueLine._line = line;
		nAncientDialogueLine._ancient = ancient;
		nAncientDialogueLine._character = character;
		return nAncientDialogueLine;
	}

	public override void _Ready()
	{
		ConnectSignals();
		LocString lineText = _line.LineText;
		_character.AddDetailsTo(lineText);
		GetNode<MegaRichTextLabel>("%Text").Text = lineText.GetFormattedText();
		switch (_line.Speaker)
		{
		case AncientDialogueSpeaker.Ancient:
			SetAncientAsSpeaker();
			break;
		case AncientDialogueSpeaker.Character:
			SetCharacterAsSpeaker();
			break;
		default:
			throw new ArgumentOutOfRangeException();
		}
		base.Modulate = StsColors.transparentWhite;
		_tween?.Kill();
		_tween = CreateTween();
		_tween.TweenProperty(this, "modulate:a", 1f, 0.1);
	}

	protected override void OnUnfocus()
	{
		_tween?.Kill();
		_tween = CreateTween();
		_tween.TweenProperty(this, "modulate:a", _targetAlpha, 0.1);
	}

	protected override void OnFocus()
	{
		_tween?.Kill();
		_tween = CreateTween();
		_tween.TweenProperty(this, "modulate:a", 1f, 0.1);
	}

	protected override void OnPress()
	{
	}

	public void PlaySfx()
	{
		SfxCmd.Play(_line.GetSfxOrFallbackPath());
	}

	private void SetAncientAsSpeaker()
	{
		Control node = GetNode<Control>("%AncientIcon");
		node.GetNode<TextureRect>("Icon").Texture = _ancient.RunHistoryIcon;
		node.GetNode<TextureRect>("Icon/Outline").Texture = _ancient.RunHistoryIconOutline;
		_iconNode = node;
		Control node2 = GetNode<Control>("%DialogueTailLeft");
		node2.Visible = true;
		MarginContainer node3 = GetNode<MarginContainer>("%TextContainer");
		node3.AddThemeConstantOverride(ThemeConstants.MarginContainer.marginLeft, 48);
		GetNode<Control>("%Bubble").SelfModulate = _ancient.DialogueColor;
		node2.SelfModulate = _ancient.DialogueColor;
	}

	private void SetCharacterAsSpeaker()
	{
		Control node = GetNode<Control>("%CharacterIcon");
		node.GetNode<TextureRect>("Icon").Texture = _character.IconTexture;
		node.GetNode<TextureRect>("Icon/Outline").Texture = _character.IconOutlineTexture;
		_iconNode = node;
		Control node2 = GetNode<Control>("%DialogueTailRight");
		node2.Visible = true;
		MarginContainer node3 = GetNode<MarginContainer>("%TextContainer");
		node3.AddThemeConstantOverride(ThemeConstants.MarginContainer.marginRight, 46);
		GetNode<Control>("%Bubble").SelfModulate = _character.DialogueColor;
		node2.SelfModulate = _character.DialogueColor;
	}

	public void SetSpeakerIconVisible()
	{
		_iconNode.Visible = true;
	}

	public void SetTransparency(float alpha)
	{
		_targetAlpha = alpha;
		base.Modulate = new Color(1f, 1f, 1f, alpha);
	}
}
