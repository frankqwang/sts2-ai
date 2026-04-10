using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Models.Potions;
using MegaCrit.Sts2.Core.Models.Relics;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Nodes.Screens.Shops;
using MegaCrit.Sts2.Core.Nodes.Vfx;
using MegaCrit.Sts2.Core.Random;

namespace MegaCrit.Sts2.Core.Nodes.Events.Custom;

public partial class NFakeMerchant : Control, ICustomEventNode, IScreenContext
{
	private const float _animVariance = 0.5f;

	private readonly List<Player> _players = new List<Player>();

	private FakeMerchant _event;

	private MerchantDialogueSet _dialogue;

	private NProceedButton _proceedButton;

	private Control _characterContainer;

	private Control _inputBlocker;

	private NMerchantInventory _inventory;

	public NMerchantButton MerchantButton { get; private set; }

	public IScreenContext CurrentScreenContext
	{
		get
		{
			if (!_inventory.IsOpen)
			{
				return this;
			}
			return _inventory;
		}
	}

	public Control? DefaultFocusedControl => null;

	public void Initialize(EventModel eventModel)
	{
		_event = (FakeMerchant)eventModel;
		_dialogue = FakeMerchant.Dialogue;
		_players.AddRange(_event.Owner.RunState.Players);
	}

	public override void _Ready()
	{
		_proceedButton = GetNode<NProceedButton>("%ProceedButton");
		_proceedButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(HideScreen));
		_proceedButton.UpdateText(NProceedButton.ProceedLoc);
		_proceedButton.SetPulseState(isPulsing: false);
		_proceedButton.Enable();
		MerchantButton = GetNode<NMerchantButton>("%MerchantButton");
		if (_event.StartedFight)
		{
			MerchantButton.Hide();
			Player me = LocalContext.GetMe(_players);
			if (me.GetRelic<FakeMerchantsRug>() != null)
			{
				MegaSprite megaSprite = new MegaSprite(GetNode<Node2D>("%FakeMerchantBackground"));
				megaSprite.GetSkeleton().FindBone("rug").Hide();
			}
		}
		else
		{
			MerchantButton.IsLocalPlayerDead = LocalContext.GetMe(_players).Creature.IsDead;
			MerchantButton.PlayerDeadLines = _dialogue.PlayerDeadLines;
			MerchantButton.Connect(NMerchantButton.SignalName.MerchantOpened, Callable.From<NMerchantButton>(OnMerchantOpened));
		}
		_inventory = GetNode<NMerchantInventory>("%Inventory");
		_inventory.MouseFilter = MouseFilterEnum.Ignore;
		_inventory.Initialize(_event.Inventory, _dialogue);
		_characterContainer = GetNode<Control>("%CharacterContainer");
		_inputBlocker = GetNode<Control>("%InputBlocker");
		NMapScreen.Instance.SetTravelEnabled(enabled: true);
		NGame.Instance.SetScreenShakeTarget(this);
		AfterRoomIsLoaded();
	}

	public override void _EnterTree()
	{
		ActiveScreenContext.Instance.Updated += OnActiveScreenUpdated;
		NMapScreen.Instance.Connect(NMapScreen.SignalName.Opened, Callable.From(ToggleMerchantTrack));
		NMapScreen.Instance.Connect(NMapScreen.SignalName.Closed, Callable.From(ToggleMerchantTrack));
	}

	public override void _ExitTree()
	{
		NGame.Instance.ClearScreenShakeTarget();
		ActiveScreenContext.Instance.Updated -= OnActiveScreenUpdated;
		NMapScreen.Instance.Disconnect(NMapScreen.SignalName.Opened, Callable.From(ToggleMerchantTrack));
		NMapScreen.Instance.Disconnect(NMapScreen.SignalName.Closed, Callable.From(ToggleMerchantTrack));
	}

	public async Task FoulPotionThrown(FoulPotion potion)
	{
		LocString locString = Rng.Chaotic.NextItem(_dialogue.FoulPotionLines);
		if (locString != null)
		{
			NSpeechBubbleVfx nSpeechBubbleVfx = MerchantButton.PlayDialogue(locString);
			if (nSpeechBubbleVfx != null)
			{
				await Cmd.Wait((float)nSpeechBubbleVfx.SecondsToDisplay - 1f);
			}
		}
	}

	private void ToggleMerchantTrack()
	{
	}

	private void AfterRoomIsLoaded()
	{
		Player me = LocalContext.GetMe(_players);
		_players.Remove(me);
		_players.Insert(0, me);
		int num = Mathf.CeilToInt(Mathf.Sqrt(_players.Count));
		for (int i = 0; i < num; i++)
		{
			float num2 = -75f * (float)i;
			for (int j = 0; j < num; j++)
			{
				int num3 = i * num + j;
				if (num3 >= _players.Count)
				{
					break;
				}
				NCreatureVisuals nCreatureVisuals = _players[num3].Character.CreateVisuals();
				_characterContainer.AddChildSafely(nCreatureVisuals);
				StartCharacterAnimation(nCreatureVisuals);
				_characterContainer.MoveChild(nCreatureVisuals, 0);
				nCreatureVisuals.Position = new Vector2(num2, -50f * (float)i);
				if (i > 0)
				{
					nCreatureVisuals.Modulate = new Color(0.5f, 0.5f, 0.5f);
				}
				num2 -= nCreatureVisuals.Bounds.Size.X * 0.5f + 25f;
			}
		}
		if (!_event.StartedFight)
		{
			TaskHelper.RunSafely(ShowWelcomeDialogue());
		}
	}

	private async Task ShowWelcomeDialogue()
	{
		LocString line = Rng.Chaotic.NextItem(_dialogue.WelcomeLines);
		if (line != null)
		{
			await Cmd.Wait(0.75f);
			SfxCmd.Play("event:/sfx/npcs/reverse_merchant/reverse_merchant_laugh");
			MerchantButton.PlayDialogue(line, 4.0);
		}
	}

	private void StartCharacterAnimation(NCreatureVisuals visuals)
	{
		MegaTrackEntry megaTrackEntry = visuals.SpineBody.GetAnimationState().SetAnimation("relaxed_loop");
		if (megaTrackEntry != null)
		{
			megaTrackEntry.SetLoop(loop: true);
			megaTrackEntry.SetTimeScale(Rng.Chaotic.NextFloat(0.9f, 1.1f));
			float animationEnd = megaTrackEntry.GetAnimationEnd();
			megaTrackEntry.SetTrackTime((animationEnd + Rng.Chaotic.NextFloat(-0.5f, 0.5f)) % animationEnd);
		}
	}

	private void HideScreen(NButton _)
	{
		NMapScreen.Instance.Open();
	}

	private void OnMerchantOpened(NMerchantButton _)
	{
		OpenInventory();
	}

	private void OpenInventory()
	{
		if (!_inventory.IsOpen)
		{
			_proceedButton.Disable();
			_inventory.Open();
			MerchantButton.Disable();
			_inventory.Connect(NMerchantInventory.SignalName.InventoryClosed, Callable.From(delegate
			{
				MerchantButton.Enable();
				ShowProceedButton();
			}), 4u);
		}
	}

	private void ShowProceedButton()
	{
		_proceedButton.Enable();
		_proceedButton.SetPulseState(isPulsing: true);
	}

	private void OnActiveScreenUpdated()
	{
		this.UpdateControllerNavEnabled();
		if (ActiveScreenContext.Instance.IsCurrent(this))
		{
			MerchantButton.Enable();
			if (!_proceedButton.IsEnabled)
			{
				_proceedButton.Enable();
			}
		}
		else
		{
			MerchantButton.Disable();
			_proceedButton.Disable();
		}
	}

	public void BlockInput()
	{
		_inputBlocker.MouseFilter = MouseFilterEnum.Stop;
		NHotkeyManager.Instance.AddBlockingScreen(_inputBlocker);
	}

	public void UnblockInput()
	{
		_inputBlocker.MouseFilter = MouseFilterEnum.Ignore;
		NHotkeyManager.Instance.RemoveBlockingScreen(_inputBlocker);
	}
}
