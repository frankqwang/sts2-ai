using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Platform;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Screens.RunHistoryScreen;

public partial class NRunHistoryPlayerIcon : NClickableControl
{
	public static readonly string scenePath = SceneHelper.GetScenePath("screens/run_history_screen/run_history_player_icon");

	private readonly List<IHoverTip> _hoverTips = new List<IHoverTip>();

	private Control _ascensionIcon;

	private MegaLabel _ascensionLabel;

	private NSelectionReticle _selectionReticle;

	private Control? _currentIcon;

	public RunHistoryPlayer Player { get; private set; }

	public override void _Ready()
	{
		_ascensionIcon = GetNode<Control>("%AscensionIcon");
		_ascensionLabel = GetNode<MegaLabel>("%AscensionLabel");
		_selectionReticle = GetNode<NSelectionReticle>("%SelectionReticle");
		ConnectSignals();
	}

	public void LoadRun(RunHistoryPlayer player, RunHistory history)
	{
		Player = player;
		CharacterModel byId = ModelDb.GetById<CharacterModel>(player.Character);
		_currentIcon?.QueueFreeSafely();
		_currentIcon = byId.Icon;
		this.AddChildSafely(_currentIcon);
		MoveChild(_currentIcon, 0);
		LocString locString = new LocString("ascension", "PORTRAIT_TITLE");
		locString.Add("character", byId.Title);
		locString.Add("ascension", history.Ascension);
		LocString locString2 = new LocString("ascension", "PORTRAIT_DESCRIPTION");
		List<string> list = new List<string>();
		for (int i = 1; i <= history.Ascension; i++)
		{
			list.Add(AscensionHelper.GetTitle(i).GetFormattedText());
		}
		locString2.Add("ascensions", list);
		_selectionReticle.Visible = history.Players.Count > 1;
		_ascensionIcon.Visible = false;
		_ascensionLabel.SetTextAutoSize((history.Ascension > 0) ? history.Ascension.ToString() : string.Empty);
		LocString locString3 = new LocString("run_history", "PLAYER_HOVER");
		if (history.Players.Count > 1)
		{
			locString3.Add("PlayerName", PlatformUtil.GetPlayerName(history.PlatformType, player.Id));
			locString3.Add("CharacterName", byId.Title.GetFormattedText());
		}
		else
		{
			locString3.Add("PlayerName", byId.Title.GetFormattedText());
			locString3.Add("CharacterName", string.Empty);
		}
		_hoverTips.Add(new HoverTip(locString3));
		if (history.Ascension > 0)
		{
			_hoverTips.Add(AscensionHelper.GetHoverTip(byId, history.Ascension));
		}
	}

	public void Select()
	{
		_ascensionIcon.Visible = _ascensionLabel.Text != string.Empty;
		_selectionReticle.OnSelect();
	}

	public void Deselect()
	{
		_ascensionIcon.Visible = false;
		_selectionReticle.OnDeselect();
	}

	protected override void OnFocus()
	{
		NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, _hoverTips);
		nHoverTipSet.GlobalPosition = base.GlobalPosition + new Vector2(0f, base.Size.Y + 20f);
	}

	protected override void OnUnfocus()
	{
		NHoverTipSet.Remove(this);
	}
}
