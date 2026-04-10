using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Bestiary;

public partial class NBestiary : NSubmenu
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/bestiary/bestiary");

	private MegaRichTextLabel _monsterNameLabel;

	private MegaLabel _epithet;

	private NScrollableContainer _sidebar;

	private VBoxContainer _bestiaryList;

	private static readonly LocString _locked = new LocString("bestiary", "LOCKED.monsterTitle");

	private Control _monsterVisualsContainer;

	private static readonly LocString _placeholderDesc = new LocString("bestiary", "DESCRIPTION.placeholder");

	private MegaRichTextLabel _descriptionLabel;

	private Control _selectionArrow;

	private Control _moveList;

	private Control _moveContainer;

	private Tween? _arrowTween;

	private static readonly Vector2 _arrowOffset = new Vector2(-34f, 4f);

	private bool _arrowPosReset = true;

	private NCreatureVisuals? _monsterVisuals;

	private MegaSprite? _animController;

	private MegaAnimationState? _animState;

	private NBestiaryEntry? _selectedEntry;

	private Tween? _tween;

	private bool _isPlayingMoveAnim;

	protected override Control? InitialFocusedControl => _bestiaryList.GetChildren().OfType<NBestiaryEntry>().FirstOrDefault();

	public static string[] AssetPaths
	{
		get
		{
			List<string> list = new List<string>();
			list.Add(_scenePath);
			list.AddRange(NBestiaryEntry.AssetPaths);
			return list.ToArray();
		}
	}

	public static NBestiary? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NBestiary>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		ConnectSignals();
		GetNode<MegaLabel>("%MoveHeader").SetTextAutoSize(new LocString("bestiary", "ACTIONS.header").GetFormattedText());
		_sidebar = GetNode<NScrollableContainer>("%Sidebar");
		_bestiaryList = GetNode<VBoxContainer>("%BestiaryList");
		_monsterNameLabel = GetNode<MegaRichTextLabel>("%MonsterName");
		_epithet = GetNode<MegaLabel>("%Epithet");
		_descriptionLabel = GetNode<MegaRichTextLabel>("%Description");
		_moveContainer = GetNode<Control>("%MoveContainer");
		_selectionArrow = GetNode<Control>("%SelectionArrow");
		_monsterVisualsContainer = GetNode<Control>("%MonsterVisualsContainer");
		_moveList = GetNode<Control>("%MoveList");
	}

	public override void OnSubmenuOpened()
	{
		CreateEntries();
	}

	public override void OnSubmenuClosed()
	{
		_selectedEntry = null;
		_monsterVisuals?.QueueFreeSafely();
		_monsterVisuals = null;
		foreach (Node child in _bestiaryList.GetChildren())
		{
			child.QueueFreeSafely();
		}
	}

	private void CreateEntries()
	{
		HashSet<ModelId> hashSet = (from e in SaveManager.Instance.Progress.EnemyStats.Values
			where e.TotalWins > 0
			select e.Id).ToHashSet();
		foreach (MonsterModel item in ModelDb.Monsters.OrderBy((MonsterModel m) => m.Id.Entry))
		{
			bool flag = hashSet.Contains(item.Id);
			NBestiaryEntry nBestiaryEntry = NBestiaryEntry.Create(item, !flag);
			_bestiaryList.AddChildSafely(nBestiaryEntry);
			nBestiaryEntry.Connect(NClickableControl.SignalName.Released, Callable.From<NBestiaryEntry>(OnMonsterClicked));
		}
		_sidebar.InstantlyScrollToTop();
		SelectMonster(_bestiaryList.GetChild<NBestiaryEntry>(0));
	}

	private void OnMonsterClicked(NBestiaryEntry entry)
	{
		SelectMonster(entry);
	}

	private void SelectMonster(NBestiaryEntry entry)
	{
		if (entry == _selectedEntry)
		{
			return;
		}
		_moveList.FreeChildren();
		_arrowPosReset = true;
		_selectedEntry?.Deselect();
		_selectedEntry = entry;
		_selectedEntry.Select();
		MonsterModel monster = _selectedEntry.Monster;
		if (entry.IsLocked)
		{
			_monsterNameLabel.Text = _locked.GetFormattedText();
			_descriptionLabel.Text = _placeholderDesc.GetFormattedText();
			_monsterVisuals?.QueueFreeSafely();
			_monsterVisuals = null;
			_moveContainer.Visible = false;
			return;
		}
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_descriptionLabel.Text = _placeholderDesc.GetFormattedText();
		_descriptionLabel.Modulate = StsColors.transparentWhite;
		_monsterNameLabel.Text = monster.Title.GetFormattedText();
		_monsterNameLabel.SelfModulate = StsColors.transparentWhite;
		_epithet.Modulate = StsColors.transparentWhite;
		_moveContainer.Modulate = StsColors.transparentWhite;
		_tween.TweenProperty(_monsterNameLabel, "position:y", 88f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
			.From(24f);
		_tween.TweenProperty(_monsterNameLabel, "self_modulate:a", 1f, 0.5);
		_tween.TweenProperty(_epithet, "modulate:a", 1f, 0.5).SetDelay(0.2);
		_tween.TweenProperty(_descriptionLabel, "position:y", 894f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
			.From(958f);
		_tween.TweenProperty(_descriptionLabel, "modulate:a", 1f, 0.5);
		_tween.TweenProperty(_moveContainer, "position:x", 242f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
			.From(210f)
			.SetDelay(0.2);
		_tween.TweenProperty(_moveContainer, "modulate:a", 1f, 0.5).SetDelay(0.2);
		_monsterVisuals?.QueueFreeSafely();
		_monsterVisuals = monster.CreateVisuals();
		_monsterVisualsContainer.AddChildSafely(_monsterVisuals);
		_monsterVisuals.Position = new Vector2(0f, _monsterVisuals.Bounds.Size.Y * 0.5f);
		_monsterVisuals.Modulate = StsColors.transparentBlack;
		_tween.TweenProperty(_monsterVisuals, "modulate", Colors.White, 0.25);
		_isPlayingMoveAnim = false;
		if (_monsterVisuals.HasSpineAnimation)
		{
			_moveContainer.Visible = true;
			_animController = _monsterVisuals.SpineBody;
			_animState = _animController.GetAnimationState();
			monster.GenerateAnimator(_animController);
			_monsterVisuals.SetUpSkin(monster);
			PlayIdleAnim();
			{
				foreach (BestiaryMonsterMove item in monster.MonsterMoveList(_monsterVisuals))
				{
					NBestiaryMoveButton nBestiaryMoveButton = NBestiaryMoveButton.Create(item);
					_moveList.AddChildSafely(nBestiaryMoveButton);
					nBestiaryMoveButton.Connect(NClickableControl.SignalName.Focused, Callable.From<NBestiaryMoveButton>(OnMoveButtonFocused));
					nBestiaryMoveButton.Connect(NClickableControl.SignalName.Unfocused, Callable.From<NBestiaryMoveButton>(OnMoveButtonUnfocused));
					nBestiaryMoveButton.Connect(NClickableControl.SignalName.Released, Callable.From<NBestiaryMoveButton>(OnMoveButtonClicked));
				}
				return;
			}
		}
		_moveContainer.Visible = false;
	}

	private void OnMoveButtonFocused(NBestiaryMoveButton button)
	{
		if (_arrowPosReset)
		{
			_selectionArrow.GlobalPosition = button.GlobalPosition + _arrowOffset;
			_arrowPosReset = false;
		}
		_arrowTween?.Kill();
		_arrowTween = CreateTween().SetParallel();
		_arrowTween.TweenProperty(_selectionArrow, "modulate:a", 1f, 0.05);
		_arrowTween.TweenProperty(_selectionArrow, "global_position", button.GlobalPosition + _arrowOffset, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	private void OnMoveButtonUnfocused(NBestiaryMoveButton button)
	{
		_arrowTween?.Kill();
		_arrowTween = CreateTween().SetParallel();
		_arrowTween.TweenProperty(_selectionArrow, "modulate:a", 0f, 0.25);
	}

	private void OnMoveButtonClicked(NButton button)
	{
		NBestiaryMoveButton nBestiaryMoveButton = (NBestiaryMoveButton)button;
		PlayMoveAnim(nBestiaryMoveButton.Move.animId);
		nBestiaryMoveButton.PlaySfx();
	}

	private void PlayIdleAnim()
	{
		if (_monsterVisuals != null && _monsterVisuals.HasSpineAnimation)
		{
			_isPlayingMoveAnim = false;
			_animState.SetAnimation("idle_loop");
		}
	}

	private void PlayMoveAnim(string animId)
	{
		if (_monsterVisuals != null && _monsterVisuals.HasSpineAnimation)
		{
			_animState.SetAnimation(animId, loop: false);
			if (!_isPlayingMoveAnim)
			{
				_animController.ConnectAnimationCompleted(Callable.From<GodotObject, GodotObject, GodotObject>(OnMoveAnimCompleted));
			}
			_isPlayingMoveAnim = true;
		}
	}

	private void OnMoveAnimCompleted(GodotObject _, GodotObject __, GodotObject ___)
	{
		if (_isPlayingMoveAnim)
		{
			_isPlayingMoveAnim = false;
			_animController.DisconnectAnimationCompleted(Callable.From<GodotObject, GodotObject, GodotObject>(OnMoveAnimCompleted));
			PlayIdleAnim();
		}
	}
}
