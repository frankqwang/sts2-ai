using System;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Ftue;

public partial class NMapSelectFtue : NFtue
{
	public const string id = "map_select_ftue";

	private static readonly string _scenePath = SceneHelper.GetScenePath("ftue/map_select_ftue");

	private NButton _confirmButton;

	private MegaLabel _header;

	private MegaRichTextLabel _description;

	private Vector2 _ftuePosition;

	private Control _firstNode;

	private int _defaultZIndex;

	public override void _Ready()
	{
		_header = GetNode<MegaLabel>("%Header");
		_header.SetTextAutoSize(new LocString("ftues", "MAP_SELECT_TITLE").GetFormattedText());
		_description = GetNode<MegaRichTextLabel>("%Description");
		_description.Text = new LocString("ftues", "MAP_SELECT_DESCRIPTION").GetFormattedText();
		_confirmButton = GetNode<NButton>("%FtueConfirmButton");
		_confirmButton.Connect(NClickableControl.SignalName.Released, Callable.From((Action<NButton>)CloseFtue));
		GetNode<Control>("%Positioner").GlobalPosition = _ftuePosition;
		_defaultZIndex = _firstNode.ZIndex;
		_firstNode.ZIndex++;
	}

	public static NMapSelectFtue Create(Control roomNode)
	{
		NMapSelectFtue nMapSelectFtue = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NMapSelectFtue>(PackedScene.GenEditState.Disabled);
		nMapSelectFtue._ftuePosition = roomNode.GlobalPosition + roomNode.Size * 0.5f;
		nMapSelectFtue._firstNode = roomNode;
		return nMapSelectFtue;
	}

	private void CloseFtue(NButton _)
	{
		_firstNode.ZIndex = _defaultZIndex;
		CloseFtue();
	}

	public async Task WaitForPlayerToConfirm()
	{
		await ToSignal(_confirmButton, NClickableControl.SignalName.Released);
	}
}
