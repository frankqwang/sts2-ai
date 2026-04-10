using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models.Potions;
using MegaCrit.Sts2.Core.Nodes.Audio;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Ftue;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Nodes.Screens.Shops;
using MegaCrit.Sts2.Core.Nodes.Vfx;
using MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Rooms;

public partial class NMerchantRoom : Control, IScreenContext, IRoomWithProceedButton
{
	private const float _animVariance = 0.5f;

	private static readonly string _scenePath = SceneHelper.GetScenePath("rooms/merchant_room");

	private readonly List<Player> _players = new List<Player>();

	private MerchantDialogueSet _dialogue;

	private NProceedButton _proceedButton;

	private Control _characterContainer;

	private Control _inputBlocker;

	private readonly List<NMerchantCharacter> _playerVisuals = new List<NMerchantCharacter>();

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public static NMerchantRoom? Instance => NRun.Instance?.MerchantRoom;

	public NProceedButton ProceedButton => _proceedButton;

	public MerchantRoom Room { get; private set; }

	public NMerchantInventory Inventory { get; private set; }

	public NMerchantButton MerchantButton { get; private set; }

	public IReadOnlyList<NMerchantCharacter> PlayerVisuals => _playerVisuals;

	public Control? DefaultFocusedControl => null;

	public static NMerchantRoom? Create(MerchantRoom room, IReadOnlyList<Player> players)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NMerchantRoom nMerchantRoom = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NMerchantRoom>(PackedScene.GenEditState.Disabled);
		nMerchantRoom.Room = room;
		nMerchantRoom._players.AddRange(players);
		nMerchantRoom._dialogue = MerchantRoom.Dialogue;
		return nMerchantRoom;
	}

	public override void _Ready()
	{
		_proceedButton = GetNode<NProceedButton>("%ProceedButton");
		_proceedButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(HideScreen));
		_proceedButton.UpdateText(NProceedButton.ProceedLoc);
		_proceedButton.SetPulseState(isPulsing: false);
		_proceedButton.Enable();
		MerchantButton = GetNode<NMerchantButton>("%MerchantButton");
		MerchantButton.IsLocalPlayerDead = LocalContext.GetMe(_players).Creature.IsDead;
		MerchantButton.PlayerDeadLines = _dialogue.PlayerDeadLines;
		MerchantButton.Connect(NMerchantButton.SignalName.MerchantOpened, Callable.From<NMerchantButton>(OnMerchantOpened));
		Inventory = GetNode<NMerchantInventory>("%Inventory");
		Inventory.MouseFilter = MouseFilterEnum.Ignore;
		Inventory.Initialize(Room.Inventory, _dialogue);
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

	public void FoulPotionThrown(FoulPotion potion)
	{
		SfxCmd.Play("event:/sfx/npcs/merchant/merchant_thank_yous");
		LocString locString = Rng.Chaotic.NextItem(_dialogue.FoulPotionLines);
		if (locString != null)
		{
			NSpeechBubbleVfx nSpeechBubbleVfx = MerchantButton.PlayDialogue(locString);
			if (nSpeechBubbleVfx != null)
			{
				NGame.Instance?.ScreenRumble(ShakeStrength.Medium, ShakeDuration.Short, RumbleStyle.Rumble);
			}
		}
	}

	private void ToggleMerchantTrack()
	{
		NRunMusicController.Instance?.ToggleMerchantTrack();
	}

	private void AfterRoomIsLoaded()
	{
		Player me = LocalContext.GetMe(_players);
		_players.Remove(me);
		_players.Insert(0, me);
		int num = Mathf.CeilToInt(Mathf.Sqrt(_players.Count));
		for (int i = 0; i < num; i++)
		{
			float num2 = -140f * (float)i;
			for (int j = 0; j < num; j++)
			{
				int num3 = i * num + j;
				if (num3 >= _players.Count)
				{
					break;
				}
				NMerchantCharacter nMerchantCharacter = PreloadManager.Cache.GetScene(_players[num3].Character.MerchantAnimPath).Instantiate<NMerchantCharacter>(PackedScene.GenEditState.Disabled);
				_characterContainer.AddChildSafely(nMerchantCharacter);
				_characterContainer.MoveChild(nMerchantCharacter, 0);
				nMerchantCharacter.Position = new Vector2(num2, -50f * (float)i);
				if (i > 0)
				{
					nMerchantCharacter.Modulate = new Color(0.5f, 0.5f, 0.5f);
				}
				num2 -= 275f;
				_playerVisuals.Add(nMerchantCharacter);
			}
		}
	}

	private void HideScreen(NButton _)
	{
		if (!MerchantFtueCheck())
		{
			NMapScreen.Instance.Open();
		}
	}

	private bool MerchantFtueCheck()
	{
		if (!SaveManager.Instance.SeenFtue("merchant_ftue"))
		{
			NModalContainer.Instance.Add(NMerchantFtue.Create(this));
			SaveManager.Instance.MarkFtueAsComplete("merchant_ftue");
			return true;
		}
		return false;
	}

	private void OnMerchantOpened(NMerchantButton _)
	{
		OpenInventory();
	}

	public void OpenInventory()
	{
		if (!Inventory.IsOpen)
		{
			_proceedButton.Disable();
			Inventory.Open();
			MerchantButton.Disable();
			Inventory.Connect(NMerchantInventory.SignalName.InventoryClosed, Callable.From(delegate
			{
				MerchantButton.Enable();
				_proceedButton.Enable();
				_proceedButton.SetPulseState(isPulsing: true);
			}), 4u);
		}
	}

	private void OnActiveScreenUpdated()
	{
		this.UpdateControllerNavEnabled();
		if (ActiveScreenContext.Instance.IsCurrent(this))
		{
			MerchantButton.Enable();
			_proceedButton.Enable();
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
