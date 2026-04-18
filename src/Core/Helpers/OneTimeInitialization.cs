using System;
using System.Runtime.CompilerServices;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;
using SmartFormat.Extensions;

namespace MegaCrit.Sts2.Core.Helpers;

public static class OneTimeInitialization
{
	private enum State
	{
		None,
		VeryEarly,
		Essential,
		Done
	}

	private static AtlasResourceLoader? _atlasResourceLoader;

	private static State _state;

	public static ReadSaveResult<SettingsSave> SettingsReadResult { get; private set; }

	public static void ExecuteVeryEarly()
	{
		if (_state != State.None)
		{
			Log.Error($"Tried to call OneTimeInitialization.ExecuteVeryEarly in state {_state}!");
			return;
		}
		_state = State.VeryEarly;
		if (TestMode.IsOn)
		{
			SettingsReadResult = SaveManager.Instance.InitSettingsDataForTest();
		}
		else
		{
			SettingsReadResult = SaveManager.Instance.InitSettingsData();
		}
		ModManager.Initialize(new ModManagerFileIo(), SaveManager.Instance.SettingsSave.ModSettings);
		UserDataPathProvider.IsRunningModded = ModManager.IsRunningModded();
	}

	public static void ExecuteEssential()
	{
		if (_state != State.VeryEarly)
		{
			Log.Error($"Tried to call OneTimeInitialization.ExecuteEssential in state {_state}!");
			return;
		}
		_state = State.Essential;
		_atlasResourceLoader = new AtlasResourceLoader();
		ResourceLoader.AddResourceFormatLoader(_atlasResourceLoader, atFront: true);
		AtlasManager.LoadEssentialAtlases();
		LocManager.Initialize();
		ModelDb.Init();
		ModelIdSerializationCache.Init();
		ModelDb.InitIds();
	}

	public static void ExecuteDeferred()
	{
		if (_state != State.Essential)
		{
			Log.Error($"Tried to call OneTimeInitialization.ExecuteDeferred in state {_state}!");
			return;
		}
		_state = State.Done;
		AtlasManager.LoadAllAtlases();
		if (!OS.HasFeature("editor"))
		{
			ModelDb.Preload();
			PrewarmJit();
			ConditionalFormatter conditionalFormatter = new ConditionalFormatter();
		}
	}

	private static void PrewarmJit()
	{
		Type typeFromHandle = typeof(PacketWriter);
		Type typeFromHandle2 = typeof(PacketReader);
		foreach (Type subtype in ReflectionHelper.GetSubtypes<IPacketSerializable>())
		{
			RuntimeHelpers.PrepareMethod(subtype.GetMethod("Serialize").MethodHandle);
			RuntimeHelpers.PrepareMethod(subtype.GetMethod("Deserialize").MethodHandle);
			RuntimeHelpers.PrepareMethod(typeFromHandle.GetMethod("WriteList").MethodHandle, new RuntimeTypeHandle[1] { subtype.TypeHandle });
			RuntimeHelpers.PrepareMethod(typeFromHandle.GetMethod("Write").MethodHandle, new RuntimeTypeHandle[1] { subtype.TypeHandle });
			RuntimeHelpers.PrepareMethod(typeFromHandle2.GetMethod("ReadList").MethodHandle, new RuntimeTypeHandle[1] { subtype.TypeHandle });
			RuntimeHelpers.PrepareMethod(typeFromHandle2.GetMethod("Read").MethodHandle, new RuntimeTypeHandle[1] { subtype.TypeHandle });
		}
	}
}
