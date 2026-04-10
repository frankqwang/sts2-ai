using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Characters;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Runs.History;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;

namespace MegaCrit.Sts2.Core.Nodes.Screens.RunHistoryScreen;

public partial class NMapPointHistoryEntry : NClickableControl
{
	private Vector2 _baseScale = Vector2.One * 0.7f;

	private RunHistory _runHistory;

	private MapPointHistoryEntry _entry;

	private RunHistoryPlayer? _player;

	private TextureRect _texture;

	private TextureRect _outline;

	private TextureRect _questIcon;

	private Tween? _animateInTween;

	private Tween? _hoverTween;

	private float _baseAngle;

	private bool _hurryUp;

	private static string ScenePath => SceneHelper.GetScenePath("screens/run_history_screen/map_point_history_entry");

	public static IEnumerable<string> AssetPaths => GetAssetPaths();

	public int FloorNum { get; private set; }

	private static IEnumerable<string> GetAssetPaths()
	{
		yield return ScenePath;
		RoomType[] array = new RoomType[6]
		{
			RoomType.Monster,
			RoomType.Elite,
			RoomType.Event,
			RoomType.Shop,
			RoomType.Treasure,
			RoomType.RestSite
		};
		RoomType[] array2 = array;
		foreach (RoomType roomType in array2)
		{
			string roomIconPath = ImageHelper.GetRoomIconPath(MapPointType.Monster, roomType, null);
			if (roomIconPath != null)
			{
				yield return roomIconPath;
			}
			string roomIconOutlinePath = ImageHelper.GetRoomIconOutlinePath(MapPointType.Monster, roomType, null);
			if (roomIconOutlinePath != null)
			{
				yield return roomIconOutlinePath;
			}
		}
		RoomType[] array3 = new RoomType[4]
		{
			RoomType.Monster,
			RoomType.Elite,
			RoomType.Shop,
			RoomType.Treasure
		};
		array2 = array3;
		foreach (RoomType roomType in array2)
		{
			string roomIconPath2 = ImageHelper.GetRoomIconPath(MapPointType.Unknown, roomType, null);
			if (roomIconPath2 != null)
			{
				yield return roomIconPath2;
			}
			string roomIconOutlinePath2 = ImageHelper.GetRoomIconOutlinePath(MapPointType.Unknown, roomType, null);
			if (roomIconOutlinePath2 != null)
			{
				yield return roomIconOutlinePath2;
			}
		}
		foreach (EncounterModel encounter in ModelDb.AllEncounters.Where((EncounterModel e) => e.RoomType == RoomType.Boss))
		{
			string roomIconPath3 = ImageHelper.GetRoomIconPath(MapPointType.Boss, RoomType.Boss, encounter.Id);
			if (roomIconPath3 != null)
			{
				yield return roomIconPath3;
			}
			string roomIconOutlinePath3 = ImageHelper.GetRoomIconOutlinePath(MapPointType.Boss, RoomType.Boss, encounter.Id);
			if (roomIconOutlinePath3 != null)
			{
				yield return roomIconOutlinePath3;
			}
		}
		foreach (AncientEventModel ancient in ModelDb.AllAncients)
		{
			string roomIconPath4 = ImageHelper.GetRoomIconPath(MapPointType.Ancient, RoomType.Event, ancient.Id);
			if (roomIconPath4 != null)
			{
				yield return roomIconPath4;
			}
			string roomIconOutlinePath4 = ImageHelper.GetRoomIconOutlinePath(MapPointType.Ancient, RoomType.Event, ancient.Id);
			if (roomIconOutlinePath4 != null)
			{
				yield return roomIconOutlinePath4;
			}
		}
	}

	public override void _Ready()
	{
		_baseAngle = Rng.Chaotic.NextGaussianFloat(0f, 1f, 0f, 5f);
		_texture = GetNode<TextureRect>("%Icon");
		_outline = GetNode<TextureRect>("%Outline");
		_questIcon = GetNode<TextureRect>("%QuestIcon");
		_texture.RotationDegrees = (Rng.Chaotic.NextBool() ? _baseAngle : (0f - _baseAngle));
		MapPointType mapPointType = _entry.MapPointType;
		RoomType roomType = _entry.Rooms.First().RoomType;
		MapPointType mapPointType2 = _entry.MapPointType;
		bool flag = (uint)(mapPointType2 - 7) <= 1u;
		string roomIconPath = ImageHelper.GetRoomIconPath(mapPointType, roomType, flag ? _entry.Rooms.First().ModelId : null);
		MapPointType mapPointType3 = _entry.MapPointType;
		RoomType roomType2 = _entry.Rooms.First().RoomType;
		mapPointType2 = _entry.MapPointType;
		flag = (uint)(mapPointType2 - 7) <= 1u;
		string roomIconOutlinePath = ImageHelper.GetRoomIconOutlinePath(mapPointType3, roomType2, flag ? _entry.Rooms.First().ModelId : null);
		if (roomIconPath != null)
		{
			_texture.Visible = true;
			_texture.Texture = PreloadManager.Cache.GetCompressedTexture2D(roomIconPath);
		}
		else
		{
			_texture.Visible = false;
		}
		if (roomIconOutlinePath != null)
		{
			_outline.Visible = true;
			_outline.Texture = PreloadManager.Cache.GetCompressedTexture2D(roomIconOutlinePath);
		}
		else
		{
			_outline.Visible = false;
		}
		_questIcon.Visible = false;
		Color modulate = base.Modulate;
		modulate.A = 1f;
		base.Modulate = modulate;
		ConnectSignals();
	}

	public static NMapPointHistoryEntry Create(RunHistory history, MapPointHistoryEntry entry, int floorNum)
	{
		NMapPointHistoryEntry nMapPointHistoryEntry = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NMapPointHistoryEntry>(PackedScene.GenEditState.Disabled);
		nMapPointHistoryEntry._runHistory = history;
		nMapPointHistoryEntry._entry = entry;
		nMapPointHistoryEntry.FloorNum = floorNum;
		return nMapPointHistoryEntry;
	}

	public void SetPlayer(RunHistoryPlayer player)
	{
		_player = player;
		_questIcon.Visible = _entry.GetEntry(_player.Id).CompletedQuests.Count > 0;
	}

	protected override void OnFocus()
	{
		if (_player == null)
		{
			throw new InvalidOperationException("Player has not been set!");
		}
		Highlight();
		HoverTipAlignment alignment = HoverTip.GetHoverTipAlignment(this);
		NHoverTipSet tip = NHoverTipSet.CreateAndShowMapPointHistory(this, NMapPointHistoryHoverTip.Create(FloorNum, _player.Id, _entry));
		Callable.From(delegate
		{
			tip.SetAlignment(this, alignment);
		}).CallDeferred();
		tip.GlobalPosition += Vector2.Down * 96f;
	}

	protected override void OnUnfocus()
	{
		Unhighlight();
		NHoverTipSet.Remove(this);
	}

	public void Highlight()
	{
		_hoverTween?.Kill();
		_hoverTween = CreateTween().SetParallel();
		_hoverTween.TweenProperty(_texture, "scale", _baseScale * 1.5f, 0.05).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_hoverTween.TweenProperty(_texture, "rotation_degrees", 0f, 0.05).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_hoverTween.TweenProperty(_outline, "modulate", StsColors.halfTransparentWhite, 0.05).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	public void Unhighlight()
	{
		_hoverTween?.Kill();
		_hoverTween = CreateTween().SetParallel();
		_hoverTween.TweenProperty(_texture, "scale", _baseScale, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_hoverTween.TweenProperty(_texture, "rotation_degrees", _baseAngle, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_hoverTween.TweenProperty(_outline, "modulate", StsColors.quarterTransparentBlack, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	public async Task AnimateIn(int index)
	{
		base.Visible = true;
		base.Scale = Vector2.One * 0.01f;
		_animateInTween?.Kill();
		_animateInTween = CreateTween().SetParallel();
		_animateInTween.TweenProperty(this, "scale", Vector2.One, _hurryUp ? 0.05 : 0.4).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Spring);
		_animateInTween.TweenProperty(this, "modulate", Colors.White, _hurryUp ? 0.05 : 0.1);
		TaskHelper.RunSafely(DoAnimateInEffects());
		if (_hurryUp)
		{
			await Cmd.Wait(0.05f);
			return;
		}
		float num = ((SaveManager.Instance.PrefsSave.FastMode == FastModeType.Fast) ? 0.4f : 0.5f);
		float to = 0.2f;
		int num2 = ((SaveManager.Instance.PrefsSave.FastMode == FastModeType.Fast) ? 5 : 9);
		float weight = (float)index / (float)(index + num2);
		float seconds = Mathf.Lerp(num, to, weight);
		await Cmd.Wait(seconds);
	}

	private async Task DoAnimateInEffects()
	{
		PlayerMapPointHistoryEntry entry = _entry.GetEntry(_player.Id);
		RoomType roomType = _entry.Rooms.Last().RoomType;
		if (_runHistory.MapPointHistory.Last().Last() == _entry && !_runHistory.Win)
		{
			SfxCmd.Play("event:/sfx/block_break");
			NDebugAudioManager.Instance?.Play("STS_DeathStinger_v4_Short_SFX.mp3", 0.75f * GetSfxVolume());
			return;
		}
		if ((uint)(roomType - 1) <= 2u)
		{
			await DoCombatAnimateInEffects(roomType);
			return;
		}
		switch (roomType)
		{
		case RoomType.Shop:
			SfxCmd.Play("event:/sfx/npcs/merchant/merchant_welcome");
			break;
		case RoomType.Treasure:
			SfxCmd.Play("event:/sfx/ui/gold/gold_2", GetSfxVolume());
			break;
		case RoomType.RestSite:
			if (entry.RestSiteChoices.Contains("SMITH"))
			{
				NGame.Instance.ScreenRumble(ShakeStrength.Medium, ShakeDuration.Short, RumbleStyle.Rumble);
				NDebugAudioManager.Instance?.Play("card_smith.mp3", GetSfxVolume(), PitchVariance.Small);
			}
			else if (entry.RestSiteChoices.Contains("HEAL"))
			{
				NDebugAudioManager.Instance?.Play("SOTE_SFX_SleepBlanket_v1.mp3", GetSfxVolume(), PitchVariance.Medium);
			}
			else if (entry.RestSiteChoices.Contains("DIG"))
			{
				NDebugAudioManager.Instance.Play("sts_sfx_shovel_v1.mp3", GetSfxVolume(), PitchVariance.Small);
			}
			else if (entry.RestSiteChoices.Contains("HATCH"))
			{
				SfxCmd.Play("event:/sfx/byrdpip/byrdpip_attack");
			}
			else if (entry.RestSiteChoices.Contains("LIFT"))
			{
				NGame.Instance?.ScreenShake(ShakeStrength.Strong, ShakeDuration.Short);
			}
			else if (!entry.RestSiteChoices.Contains("MEND"))
			{
			}
			break;
		case RoomType.Event:
			SfxCmd.Play("event:/sfx/ui/clicks/ui_hover");
			break;
		}
	}

	private async Task DoCombatAnimateInEffects(RoomType roomType)
	{
		CharacterModel byId = ModelDb.GetById<CharacterModel>(_player.Character);
		ShakeStrength? shakeStrength = null;
		switch (roomType)
		{
		case RoomType.Monster:
			shakeStrength = ShakeStrength.Weak;
			await PlaySfx(GetSmallHitSfx(byId));
			break;
		case RoomType.Elite:
			shakeStrength = ShakeStrength.Medium;
			await PlaySfx(GetBigHitSfx(byId));
			break;
		case RoomType.Boss:
			shakeStrength = ShakeStrength.Strong;
			await PlaySfx(GetBigHitSfx(byId));
			break;
		}
		if (shakeStrength.HasValue)
		{
			NGame.Instance.ScreenRumble(shakeStrength.Value, ShakeDuration.Normal, RumbleStyle.Rumble);
		}
		await Cmd.Wait(0.25f);
		foreach (ModelId monsterId in _entry.Rooms.Last().MonsterIds)
		{
			MonsterModel byId2 = ModelDb.GetById<MonsterModel>(monsterId);
			if (byId2.HasDeathSfx)
			{
				SfxCmd.Play(byId2.DeathSfx);
				await Cmd.Wait(0.25f);
			}
		}
	}

	private List<string> GetSmallHitSfx(CharacterModel character)
	{
		int num;
		Span<string> span;
		int index;
		if (!(character is Defect))
		{
			if (!(character is Ironclad))
			{
				if (!(character is Necrobinder))
				{
					if (!(character is Regent))
					{
						if (character is Silent)
						{
							num = 1;
							List<string> list = new List<string>(num);
							CollectionsMarshal.SetCount(list, num);
							span = CollectionsMarshal.AsSpan(list);
							index = 0;
							span[index] = "slash_attack.mp3";
							return list;
						}
						return new List<string>();
					}
					index = 1;
					List<string> list2 = new List<string>(index);
					CollectionsMarshal.SetCount(list2, index);
					span = CollectionsMarshal.AsSpan(list2);
					num = 0;
					span[num] = "slash_attack.mp3";
					return list2;
				}
				num = 1;
				List<string> list3 = new List<string>(num);
				CollectionsMarshal.SetCount(list3, num);
				span = CollectionsMarshal.AsSpan(list3);
				index = 0;
				span[index] = "slash_attack.mp3";
				return list3;
			}
			index = 1;
			List<string> list4 = new List<string>(index);
			CollectionsMarshal.SetCount(list4, index);
			span = CollectionsMarshal.AsSpan(list4);
			num = 0;
			span[num] = "blunt_attack.mp3";
			return list4;
		}
		num = 1;
		List<string> list5 = new List<string>(num);
		CollectionsMarshal.SetCount(list5, num);
		span = CollectionsMarshal.AsSpan(list5);
		index = 0;
		span[index] = "slash_attack.mp3";
		return list5;
	}

	private List<string> GetBigHitSfx(CharacterModel character)
	{
		int num;
		Span<string> span;
		int num2;
		if (!(character is Defect))
		{
			if (!(character is Ironclad))
			{
				if (!(character is Necrobinder))
				{
					if (!(character is Regent))
					{
						if (character is Silent)
						{
							num = 2;
							List<string> list = new List<string>(num);
							CollectionsMarshal.SetCount(list, num);
							span = CollectionsMarshal.AsSpan(list);
							num2 = 0;
							span[num2] = "dagger_throw.mp3";
							num2++;
							span[num2] = "dagger_throw.mp3";
							return list;
						}
						return new List<string>();
					}
					num2 = 1;
					List<string> list2 = new List<string>(num2);
					CollectionsMarshal.SetCount(list2, num2);
					span = CollectionsMarshal.AsSpan(list2);
					num = 0;
					span[num] = "heavy_attack.mp3";
					return list2;
				}
				num = 1;
				List<string> list3 = new List<string>(num);
				CollectionsMarshal.SetCount(list3, num);
				span = CollectionsMarshal.AsSpan(list3);
				num2 = 0;
				span[num2] = "heavy_attack.mp3";
				return list3;
			}
			num2 = 1;
			List<string> list4 = new List<string>(num2);
			CollectionsMarshal.SetCount(list4, num2);
			span = CollectionsMarshal.AsSpan(list4);
			num = 0;
			span[num] = "heavy_attack.mp3";
			return list4;
		}
		num = 1;
		List<string> list5 = new List<string>(num);
		CollectionsMarshal.SetCount(list5, num);
		span = CollectionsMarshal.AsSpan(list5);
		num2 = 0;
		span[num2] = "lightning_orb_evoke.mp3";
		return list5;
	}

	private async Task PlaySfx(List<string> sfxPaths)
	{
		for (int i = 0; i < sfxPaths.Count; i++)
		{
			string text = sfxPaths[i];
			if (text.StartsWith("event:"))
			{
				SfxCmd.Play(text, GetSfxVolume());
			}
			else
			{
				NDebugAudioManager.Instance?.Play(text, GetSfxVolume(), PitchVariance.Medium);
			}
			if (i < sfxPaths.Count - 1)
			{
				await Cmd.Wait(0.1f);
			}
		}
	}

	private float GetSfxVolume()
	{
		if (!_hurryUp)
		{
			return 1f;
		}
		return 0.5f;
	}

	public void HurryUp()
	{
		_hurryUp = true;
	}

	public void SetupForAnimation()
	{
		base.Visible = false;
		base.Modulate = StsColors.transparentBlack;
	}
}
