using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Characters;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Unlocks;

namespace MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;

public partial class NCharacterSelectButton : NButton
{
	private enum State
	{
		NotSelected,
		SelectedLocally,
		SelectedRemotely
	}

	private static readonly StringName _v = new StringName("v");

	private static readonly StringName _s = new StringName("s");

	private static readonly StringName _h = new StringName("h");

	private static readonly string _playerIconScenePath = SceneHelper.GetScenePath("screens/char_select/char_select_player_icon");

	private static readonly string _unlockedIconPath = ImageHelper.GetImagePath("packed/character_select/char_select_lock3_unlocked.png");

	private TextureRect _icon;

	private TextureRect _iconAdd;

	private TextureRect _lock;

	private Control _outlineLocal;

	private Control _outlineRemote;

	private Control _outlineMixed;

	private Control _shadow;

	private Control _playerIconContainer;

	private CharacterModel _character;

	private ShaderMaterial _hsv;

	private bool _isLocked;

	private static readonly Vector2 _hoverTipOffset = new Vector2(-90f, -180f);

	private ICharacterSelectButtonDelegate? _delegate;

	private Control? _currentOutline;

	private bool _isSelected;

	private readonly HashSet<ulong> _remoteSelectedPlayers = new HashSet<ulong>();

	private State _state;

	private static readonly Vector2 _hoverScale = Vector2.One * 1.1f;

	private Tween? _hoverTween;

	private Tween? _hsvTween;

	private const float _unhoverDuration = 0.5f;

	private const float _glowSpeed = 1.6f;

	private const float _selectedSaturation = 1f;

	private const float _selectedValue = 1.1f;

	private const float _remotelySelectedSaturation = 0.8f;

	private const float _remotelySelectedValue = 0.4f;

	private const float _notSelectedSaturation = 0.2f;

	private const float _notSelectedValue = 0.4f;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlyArray<string>(new string[2] { _playerIconScenePath, _unlockedIconPath });

	public bool IsRandom { get; private set; }

	public IReadOnlyCollection<ulong> RemoteSelectedPlayers => _remoteSelectedPlayers;

	public CharacterModel Character => _character;

	public bool IsLocked => _isLocked;

	public override void _Ready()
	{
		ConnectSignals();
		_icon = GetNode<TextureRect>("%Icon");
		_iconAdd = GetNode<TextureRect>("%IconAdd");
		_lock = GetNode<TextureRect>("%Lock");
		_outlineLocal = GetNode<Control>("%OutlineLocal");
		_outlineRemote = GetNode<Control>("%OutlineRemote");
		_outlineMixed = GetNode<Control>("%OutlineMixed");
		_shadow = GetNode<Control>("%Shadow");
		_playerIconContainer = GetNode<Control>("%PlayerIconContainer");
		_hsv = (ShaderMaterial)_icon.Material;
		_hsv.SetShaderParameter(_s, 0.2f);
		_hsv.SetShaderParameter(_v, 0.4f);
		Connect(Control.SignalName.FocusEntered, Callable.From(Select));
	}

	public void Init(CharacterModel character, ICharacterSelectButtonDelegate del)
	{
		_delegate = del;
		_character = character;
		UnlockState unlockState = SaveManager.Instance.GenerateUnlockStateFromProgress();
		if (character is RandomCharacter)
		{
			IsRandom = true;
			_isLocked = ModelDb.AllCharacters.Any((CharacterModel c) => !unlockState.Characters.Contains(c));
		}
		else
		{
			_isLocked = !unlockState.Characters.Contains(_character);
		}
		if (_isLocked)
		{
			_icon.Texture = character.CharacterSelectLockedIcon;
			_lock.Visible = true;
		}
		else
		{
			_icon.Texture = character.CharacterSelectIcon;
		}
	}

	protected override void OnFocus()
	{
		if (!_isSelected)
		{
			_hoverTween?.Kill();
			base.Scale = _hoverScale;
			_hsv.SetShaderParameter(_s, 1f);
			_hsv.SetShaderParameter(_v, 1.1f);
			if (_isLocked)
			{
				HoverTip hoverTip = new HoverTip(new LocString("main_menu_ui", "CHARACTER_SELECT.locked.title"), _character.GetUnlockText());
				NHoverTipSet.CreateAndShow(this, hoverTip).GlobalPosition = base.GlobalPosition + _hoverTipOffset;
			}
			SfxCmd.Play("event:/sfx/ui/clicks/ui_hover");
		}
	}

	protected override void OnPress()
	{
	}

	protected override void OnUnfocus()
	{
		_hoverTween?.Kill();
		_hoverTween = CreateTween().SetParallel();
		_hoverTween.TweenProperty(this, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		NHoverTipSet.Remove(this);
		AnimateSaturationToCurrentState(_hoverTween);
	}

	public override void _Process(double delta)
	{
		if (_currentOutline != null)
		{
			if (_isSelected)
			{
				float a = Mathf.Lerp(0.35f, 1f, (Mathf.Cos((float)Time.GetTicksMsec() * 0.001f * 1.6f * (float)Math.PI) + 1f) * 0.5f);
				Control? currentOutline = _currentOutline;
				Color modulate = _currentOutline.Modulate;
				modulate.A = a;
				currentOutline.Modulate = modulate;
			}
			else
			{
				Control? currentOutline2 = _currentOutline;
				Color modulate = _currentOutline.Modulate;
				modulate.A = 0.5f;
				currentOutline2.Modulate = modulate;
			}
		}
	}

	public void LockForAnimation()
	{
		_icon.Texture = _character.CharacterSelectLockedIcon;
		_lock.Visible = true;
		base.ZIndex = 1;
		_lock.Modulate = Colors.White;
		Disable();
	}

	public async Task AnimateUnlock()
	{
		GpuParticles2D chargeParticles = GetNode<GpuParticles2D>("%UnlockChargeParticles");
		chargeParticles.Emitting = true;
		float num = 1f;
		Vector2 originalLockPosition = _lock.Position;
		float timer = 0f;
		NDebugAudioManager.Instance.Play("character_unlock_charge.mp3");
		while (timer < 1f)
		{
			Vector2 vector = Vector2.Right.Rotated(Rng.Chaotic.NextFloat((float)Math.PI * 2f)) * num;
			_lock.Position = originalLockPosition + vector;
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
			timer += (float)GetProcessDeltaTime();
			num = Mathf.Lerp(1f, 5f, Ease.QuadOut(timer));
		}
		NDebugAudioManager.Instance.Play("character_unlock.mp3");
		_lock.Position = originalLockPosition;
		_lock.Texture = PreloadManager.Cache.GetTexture2D(_unlockedIconPath);
		_icon.Texture = _character.CharacterSelectIcon;
		_iconAdd.Texture = _icon.Texture;
		_iconAdd.Visible = true;
		GpuParticles2D node = GetNode<GpuParticles2D>("%UnlockParticles");
		node.Emitting = true;
		chargeParticles.Emitting = false;
		Tween tween = CreateTween();
		tween.SetParallel();
		tween.TweenProperty(_iconAdd, "scale", Vector2.One * 1.5f, 1.0);
		tween.TweenProperty(_iconAdd, "modulate:a", 0f, 1.0).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Expo);
		tween.TweenProperty(_lock, "modulate:a", 0f, 0.5).SetDelay(0.5);
		base.ZIndex = 0;
		Enable();
	}

	public void Reset()
	{
		foreach (Node child in _playerIconContainer.GetChildren())
		{
			child.QueueFreeSafely();
		}
		_remoteSelectedPlayers.Clear();
		Deselect();
	}

	public void OnRemotePlayerSelected(ulong playerId)
	{
		_remoteSelectedPlayers.Add(playerId);
		RefreshState();
	}

	public void OnRemotePlayerDeselected(ulong playerId)
	{
		_remoteSelectedPlayers.Remove(playerId);
		RefreshState();
	}

	public void Select()
	{
		if (!_isSelected)
		{
			_hoverTween?.Kill();
			_isSelected = true;
			_delegate.SelectCharacter(this, _character);
			RefreshState();
		}
	}

	public void Deselect()
	{
		_isSelected = false;
		RefreshState();
	}

	private void RefreshState()
	{
		State state = (_isSelected ? State.SelectedLocally : ((_remoteSelectedPlayers.Count > 0) ? State.SelectedRemotely : State.NotSelected));
		State state2 = _state;
		if (state2 != state)
		{
			_state = state;
			if (state2 == State.NotSelected)
			{
				_hsv.SetShaderParameter(_s, GetSaturationForCurrentState());
				_hsv.SetShaderParameter(_v, GetValueForCurrentState());
			}
			else
			{
				_hoverTween?.Kill();
				_hoverTween = CreateTween().SetParallel();
				AnimateSaturationToCurrentState(_hoverTween);
			}
		}
		RefreshOutline();
		RefreshPlayerIcons();
	}

	private float GetSaturationForCurrentState()
	{
		return _state switch
		{
			State.SelectedLocally => 1f, 
			State.SelectedRemotely => 0.8f, 
			State.NotSelected => 0.2f, 
			_ => throw new ArgumentOutOfRangeException(), 
		};
	}

	private float GetValueForCurrentState()
	{
		return _state switch
		{
			State.SelectedLocally => 1.1f, 
			State.SelectedRemotely => 0.4f, 
			State.NotSelected => 0.8f, 
			_ => throw new ArgumentOutOfRangeException(), 
		};
	}

	private void AnimateSaturationToCurrentState(Tween tween)
	{
		tween.TweenMethod(Callable.From<float>(UpdateShaderS), _hsv.GetShaderParameter(_s), GetSaturationForCurrentState(), 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		tween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), GetValueForCurrentState(), 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	private void RefreshOutline()
	{
		if (_currentOutline != null)
		{
			_currentOutline.Visible = false;
		}
		if (_isSelected && _remoteSelectedPlayers.Count > 0)
		{
			_currentOutline = _outlineMixed;
		}
		else if (_isSelected)
		{
			_currentOutline = _outlineLocal;
		}
		else if (_remoteSelectedPlayers.Count > 0)
		{
			_currentOutline = _outlineRemote;
		}
		else
		{
			_currentOutline = null;
		}
		if (_currentOutline != null)
		{
			_currentOutline.Visible = true;
		}
	}

	private void RefreshPlayerIcons()
	{
		if (_delegate != null && _delegate.Lobby.NetService.Type != NetGameType.Singleplayer)
		{
			int num = _remoteSelectedPlayers.Count + (_isSelected ? 1 : 0);
			for (int i = _playerIconContainer.GetChildCount(); i < num; i++)
			{
				TextureRect child = PreloadManager.Cache.GetScene(_playerIconScenePath).Instantiate<TextureRect>(PackedScene.GenEditState.Disabled);
				_playerIconContainer.AddChildSafely(child);
			}
			while (_playerIconContainer.GetChildCount() > num)
			{
				Control child2 = _playerIconContainer.GetChild<Control>(0);
				_playerIconContainer.RemoveChildSafely(child2);
				child2.QueueFreeSafely();
			}
			for (int j = 0; j < _playerIconContainer.GetChildCount(); j++)
			{
				Control child3 = _playerIconContainer.GetChild<Control>(j);
				child3.Modulate = ((_isSelected && j == 0) ? StsColors.gold : StsColors.blue);
			}
		}
	}

	public void DebugUnlock()
	{
		_icon.Texture = _character.CharacterSelectIcon;
		_isLocked = false;
		_lock.Visible = false;
		Enable();
	}

	public void UnlockIfPossible()
	{
		UnlockState unlockState = SaveManager.Instance.GenerateUnlockStateFromProgress();
		if (unlockState.Characters.Contains(_character))
		{
			_icon.Texture = _character.CharacterSelectIcon;
			_isLocked = false;
			_lock.Visible = false;
			Enable();
		}
	}

	private void UpdateShaderH(float value)
	{
		_hsv.SetShaderParameter(_h, value);
	}

	private void UpdateShaderS(float value)
	{
		_hsv.SetShaderParameter(_s, value);
	}

	private void UpdateShaderV(float value)
	{
		_hsv.SetShaderParameter(_v, value);
	}
}
