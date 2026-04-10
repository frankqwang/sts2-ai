using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;

public partial class NCrystalSphereScreen : Control, IOverlayScreen, IScreenContext
{
	private readonly LocString _instructionsTitleLoc = new LocString("events", "CRYSTAL_SPHERE.minigame.instructions.title");

	private readonly LocString _instructionsDescriptionLoc = new LocString("events", "CRYSTAL_SPHERE.minigame.instructions.description");

	private readonly LocString _divinationsRemainLoc = new LocString("events", "CRYSTAL_SPHERE.minigame.divinationsRemain");

	private const string _scenePath = "res://scenes/events/custom/crystal_sphere/crystal_sphere_screen.tscn";

	private CrystalSphereMinigame _entity;

	private Control _itemsContainer;

	private Control _cellContainer;

	private NDivinationButton _bigDivinationButton;

	private NDivinationButton _smallDivinationButton;

	private MegaRichTextLabel _divinationsLeftLabel;

	private NCrystalSphereMask _mask;

	private NProceedButton _proceedButton;

	private MegaRichTextLabel _instructionsTitleLabel;

	private MegaRichTextLabel _instructionsDescriptionLabel;

	private Control _instructionsContainer;

	private NCrystalSphereDialogue _dialogue;

	private Tween? _fadeTween;

	public NetScreenType ScreenType => NetScreenType.None;

	public bool UseSharedBackstop => false;

	public Control DefaultFocusedControl
	{
		get
		{
			List<NCrystalSphereCell> list = (from c in _cellContainer.GetChildren().OfType<NCrystalSphereCell>()
				where c.Entity.IsHidden
				select c).ToList();
			return list[list.Count / 2];
		}
	}

	public static NCrystalSphereScreen ShowScreen(CrystalSphereMinigame grid)
	{
		NCrystalSphereScreen nCrystalSphereScreen = PreloadManager.Cache.GetScene("res://scenes/events/custom/crystal_sphere/crystal_sphere_screen.tscn").Instantiate<NCrystalSphereScreen>(PackedScene.GenEditState.Disabled);
		nCrystalSphereScreen._entity = grid;
		NOverlayStack.Instance.Push(nCrystalSphereScreen);
		return nCrystalSphereScreen;
	}

	public override void _Ready()
	{
		_itemsContainer = GetNode<Control>("%Items");
		_cellContainer = GetNode<Control>("%Cells");
		_bigDivinationButton = GetNode<NDivinationButton>("%BigDivinationButton");
		_smallDivinationButton = GetNode<NDivinationButton>("%SmallDivinationButton");
		_bigDivinationButton.SetLabel(new LocString("events", "CRYSTAL_SPHERE.button.DIVINATION_LABEL_BIG"));
		_smallDivinationButton.SetLabel(new LocString("events", "CRYSTAL_SPHERE.button.DIVINATION_LABEL_SMALL"));
		_divinationsLeftLabel = GetNode<MegaRichTextLabel>("%DivinationsLeft");
		_mask = GetNode<NCrystalSphereMask>("%ScryMask");
		_proceedButton = GetNode<NProceedButton>("%ProceedButton");
		_instructionsTitleLabel = GetNode<MegaRichTextLabel>("%InstructionsTitle");
		_instructionsDescriptionLabel = GetNode<MegaRichTextLabel>("%InstructionsDescription");
		_instructionsContainer = GetNode<Control>("%Instructions");
		_instructionsTitleLabel.SetTextAutoSize(_instructionsTitleLoc.GetFormattedText());
		_instructionsDescriptionLabel.SetTextAutoSize(_instructionsDescriptionLoc.GetFormattedText());
		_dialogue = GetNode<NCrystalSphereDialogue>("%Dialogue");
		Vector2 vector = Vector2.One * -(57 * _entity.GridSize.X) * 0.5f;
		NCrystalSphereCell[,] array = new NCrystalSphereCell[_entity.GridSize.X, _entity.GridSize.Y];
		for (int i = 0; i < _entity.GridSize.X; i++)
		{
			for (int j = 0; j < _entity.GridSize.Y; j++)
			{
				NCrystalSphereCell cell = NCrystalSphereCell.Create(_entity.cells[i, j], _mask);
				_cellContainer.AddChildSafely(cell);
				array[i, j] = cell;
				cell.Position = vector + 57f * new Vector2(i, j);
				cell.Connect(Control.SignalName.MouseEntered, Callable.From(delegate
				{
					OnHoverCell(cell);
				}));
				cell.Connect(Control.SignalName.MouseExited, Callable.From(delegate
				{
					OnUnhoverCell(cell);
				}));
				cell.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
				{
					OnHoverCell(cell);
				}));
				cell.Connect(Control.SignalName.FocusExited, Callable.From(delegate
				{
					OnUnhoverCell(cell);
				}));
				cell.Connect(NClickableControl.SignalName.MouseReleased, Callable.From<InputEvent>(delegate
				{
					TaskHelper.RunSafely(OnCellClicked(cell));
				}));
				cell.Connect(NClickableControl.SignalName.Released, Callable.From<NClickableControl>(delegate
				{
					TaskHelper.RunSafely(OnCellClicked(cell));
				}));
			}
		}
		foreach (CrystalSphereItem item in _entity.Items)
		{
			NCrystalSphereItem nCrystalSphereItem = NCrystalSphereItem.Create(item);
			nCrystalSphereItem.Size = item.Size * 57;
			_itemsContainer.AddChildSafely(nCrystalSphereItem);
			nCrystalSphereItem.Position = vector + 57f * new Vector2(item.Position.X, item.Position.Y);
			item.Revealed += OnItemRevealed;
		}
		_bigDivinationButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(SetBigDivination));
		_smallDivinationButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(SetSmallDivination));
		_proceedButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnProceedButtonPressed));
		_smallDivinationButton.SetHotkeys(new string[1] { MegaInput.viewExhaustPileAndTabRight });
		_bigDivinationButton.SetHotkeys(new string[1] { MegaInput.viewDeckAndTabLeft });
		UpdateDivinationsLeft();
		_proceedButton.Disable();
		_proceedButton.UpdateText(NProceedButton.ProceedLoc);
		for (int num = 0; num < array.GetLength(0); num++)
		{
			for (int num2 = 0; num2 < array.GetLength(1); num2++)
			{
				Control control = array[num, num2];
				control.FocusNeighborTop = ((num2 > 0) ? array[num, num2 - 1].GetPath() : array[num, num2].GetPath());
				control.FocusNeighborBottom = ((num2 < array.GetLength(1) - 1) ? array[num, num2 + 1].GetPath() : array[num, num2].GetPath());
				control.FocusNeighborLeft = ((num > 0) ? array[num - 1, num2].GetPath() : array[num, num2].GetPath());
				control.FocusNeighborRight = ((num < array.GetLength(0) - 1) ? array[num + 1, num2].GetPath() : array[num, num2].GetPath());
			}
		}
	}

	private void SetBigDivination(NButton obj)
	{
		_bigDivinationButton.SetActive(isActive: true);
		_smallDivinationButton.SetActive(isActive: false);
		_entity.SetTool(CrystalSphereMinigame.CrystalSphereToolType.Big);
	}

	private void SetSmallDivination(NButton obj)
	{
		_smallDivinationButton.SetActive(isActive: true);
		_bigDivinationButton.SetActive(isActive: false);
		_entity.SetTool(CrystalSphereMinigame.CrystalSphereToolType.Small);
	}

	public override void _EnterTree()
	{
		_entity.DivinationCountChanged += UpdateDivinationsLeft;
		_entity.Finished += OnMinigameFinished;
	}

	public override void _ExitTree()
	{
		_entity.DivinationCountChanged -= UpdateDivinationsLeft;
		_entity.Finished -= OnMinigameFinished;
		_fadeTween?.Kill();
		foreach (CrystalSphereItem item in _entity.Items)
		{
			item.Revealed -= OnItemRevealed;
		}
		_entity.ForceMinigameEnd();
	}

	private void OnItemRevealed(CrystalSphereItem item)
	{
		if (item.IsGood)
		{
			_dialogue.PlayGood();
		}
		else
		{
			_dialogue.PlayBad();
		}
	}

	private async Task OnCellClicked(NCrystalSphereCell cell)
	{
		if (_entity.DivinationCount > 0)
		{
			UpdateDivinationsLeft();
			await _entity.CellClicked(cell.Entity);
			List<NCrystalSphereCell> source = (from c in _cellContainer.GetChildren().OfType<NCrystalSphereCell>()
				where c.Entity.IsHidden
				select c).ToList();
			source.OrderBy((NCrystalSphereCell c1) => new Vector2I(cell.Entity.X, cell.Entity.Y).DistanceTo(new Vector2I(c1.Entity.X, c1.Entity.Y))).First().TryGrabFocus();
		}
	}

	private void OnHoverCell(NCrystalSphereCell cell)
	{
		if (!_entity.IsFinished)
		{
			_entity.SetHoveredCell(cell.Entity);
		}
	}

	private void OnUnhoverCell(NCrystalSphereCell cell)
	{
		_entity.UnsetHoveredCell();
	}

	private void UpdateDivinationsLeft()
	{
		_divinationsRemainLoc.Add("Count", _entity.DivinationCount);
		_divinationsLeftLabel.Text = _divinationsRemainLoc.GetFormattedText() ?? "";
	}

	private void OnMinigameFinished()
	{
		_bigDivinationButton.Visible = false;
		_smallDivinationButton.Visible = false;
		_divinationsLeftLabel.Visible = false;
		_instructionsContainer.Visible = false;
		_proceedButton.Visible = true;
		_dialogue.PlayEnd();
		_proceedButton.Enable();
		NMapScreen.Instance.SetTravelEnabled(enabled: true);
	}

	private void OnProceedButtonPressed(NButton _)
	{
		TaskHelper.RunSafely(RunManager.Instance.ProceedFromTerminalRewardsScreen());
	}

	public void AfterOverlayOpened()
	{
		_itemsContainer.Visible = false;
		_fadeTween?.FastForwardToCompletion();
		_fadeTween = CreateTween();
		_fadeTween.TweenProperty(this, "modulate:a", 1.0, 0.5).From(0f);
		_fadeTween.Chain().TweenCallback(Callable.From(delegate
		{
			_dialogue.PlayStart();
			_itemsContainer.Visible = true;
		}));
	}

	public void AfterOverlayClosed()
	{
		_fadeTween?.FastForwardToCompletion();
		this.QueueFreeSafely();
	}

	public void AfterOverlayShown()
	{
		if (_entity.IsFinished)
		{
			_proceedButton.Enable();
		}
	}

	public void AfterOverlayHidden()
	{
		_proceedButton.Disable();
	}
}
