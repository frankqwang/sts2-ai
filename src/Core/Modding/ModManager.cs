using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Loader;
using System.Text.Json;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Platform.Steam;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;
using Steamworks;

namespace MegaCrit.Sts2.Core.Modding;

public static class ModManager
{
	public delegate void MetricsUploadHook(SerializableRun run, bool isVictory, ulong localPlayerId);

	private static bool _allowInitForTests;

	private static List<Mod> _mods = new List<Mod>();

	private static bool _initialized;

	private static Callback<ItemInstalled_t>? _steamItemInstalledCallback;

	private static ModSettings? _settings;

	private static IModManagerFileIo? _fileIo;

	private static readonly Dictionary<string, string> _circularDependencies = new Dictionary<string, string>();

	private static bool? _hasHarmonyPatches;

	public static IReadOnlyList<Mod> Mods => _mods;

	public static bool PlayerAgreedToModLoading => _settings?.PlayerAgreedToModLoading ?? false;

	public static event Action<Mod>? OnModDetected;

	public static event MetricsUploadHook? OnMetricsUpload;

	public static void Initialize(IModManagerFileIo fileIo, ModSettings? settings)
	{
		_settings = settings;
		_fileIo = fileIo;
		if (CommandLineHelper.HasArg("nomods"))
		{
			Log.Info("'nomods' passed as executable argument, skipping mod initialization");
		}
		else
		{
			if (TestMode.IsOn && !_allowInitForTests)
			{
				return;
			}
			_allowInitForTests = false;
			AppDomain.CurrentDomain.AssemblyResolve += HandleAssemblyResolveFailure;
			string executablePath = OS.GetExecutablePath();
			string directoryName = Path.GetDirectoryName(executablePath);
			string path = Path.Combine(directoryName, "mods");
			if (fileIo.DirectoryExists(path))
			{
				ReadModsInDirRecursive(path, ModSource.ModsDirectory, null);
			}
			if (SteamInitializer.Initialized)
			{
				ReadSteamMods();
			}
			if (_mods.Count == 0)
			{
				return;
			}
			SortModList(_settings?.ModList ?? new List<SettingsSaveMod>());
			foreach (Mod mod2 in _mods)
			{
				TryLoadMod(mod2);
			}
			if (IsRunningModded())
			{
				int value = _mods.Count((Mod m) => m.state == ModLoadState.Loaded);
				Log.Info($" --- RUNNING MODDED! --- Loaded {value} mods ({_mods.Count} total)");
			}
			_initialized = true;
			if (_settings == null)
			{
				return;
			}
			List<SettingsSaveMod> list = new List<SettingsSaveMod>();
			foreach (Mod mod in _mods)
			{
				SettingsSaveMod settingsSaveMod = new SettingsSaveMod(mod);
				bool isEnabled = _settings.ModList.FirstOrDefault((SettingsSaveMod m) => m.Id == mod.manifest?.id)?.IsEnabled ?? true;
				settingsSaveMod.IsEnabled = isEnabled;
				list.Add(settingsSaveMod);
			}
			_settings.ModList = list;
		}
	}

	public static void ResetForTests()
	{
		if (TestMode.IsOff)
		{
			throw new NotImplementedException("Tried to reset ModManager outside of tests! This is not allowed, as we cannot unload DLLs or PCKs");
		}
		_mods.Clear();
		_initialized = false;
		_settings = null;
		_fileIo = null;
		_allowInitForTests = true;
		_circularDependencies.Clear();
	}

	private static void SortModList(List<SettingsSaveMod> manualOrdering)
	{
		List<int> list = new List<int>();
		Dictionary<Mod, List<Mod>> dictionary = new Dictionary<Mod, List<Mod>>();
		for (int i = 0; i < _mods.Count; i++)
		{
			Mod mod = _mods[i];
			int num = 0;
			if (mod.manifest?.dependencies != null)
			{
				foreach (string dependencyName in mod.manifest.dependencies)
				{
					Mod mod2 = _mods.FirstOrDefault((Mod m) => m.manifest?.id == dependencyName);
					if (mod2 != null)
					{
						num++;
						if (!dictionary.TryGetValue(mod2, out var value))
						{
							value = (dictionary[mod2] = new List<Mod>());
						}
						value.Add(mod);
					}
				}
			}
			list.Add(num);
		}
		HashSet<int> seenMods = new HashSet<int>();
		List<int> currentChain = new List<int>();
		for (int num2 = 0; num2 < _mods.Count; num2++)
		{
			List<int> list3 = HasCircularDependenciesRecursive(num2, seenMods, currentChain, list);
			if (list3 == null)
			{
				continue;
			}
			string value2 = string.Join(", ", list3.Select((int index) => _mods[index].manifest?.id));
			foreach (int item in list3)
			{
				string text = _mods[item].manifest?.id;
				if (text != null)
				{
					_circularDependencies[text] = value2;
				}
			}
		}
		PriorityQueue<Mod, int> priorityQueue = new PriorityQueue<Mod, int>();
		Dictionary<string, int> dictionary2 = new Dictionary<string, int>();
		for (int num3 = 0; num3 < manualOrdering.Count; num3++)
		{
			dictionary2[manualOrdering[num3].Id] = num3;
		}
		for (int num4 = 0; num4 < _mods.Count; num4++)
		{
			Mod mod3 = _mods[num4];
			if (list[num4] == 0)
			{
				int value3;
				int priority = (dictionary2.TryGetValue(mod3.manifest.id, out value3) ? value3 : 999999999);
				priorityQueue.Enqueue(mod3, priority);
			}
		}
		if (priorityQueue.Count == 0)
		{
			Log.Error($"Detected {_mods.Count} mods, but all of them have dependencies! Something seems wrong");
		}
		List<Mod> list4 = new List<Mod>();
		while (priorityQueue.Count > 0)
		{
			Mod mod4 = priorityQueue.Dequeue();
			list4.Add(mod4);
			if (!dictionary.TryGetValue(mod4, out var value4))
			{
				continue;
			}
			foreach (Mod item2 in value4)
			{
				int num5 = _mods.IndexOf(item2);
				if (num5 < 0)
				{
					throw new InvalidOperationException("Bug in mod sorting logic!");
				}
				list[num5]--;
				if (list[num5] == 0)
				{
					int value5;
					int priority2 = (dictionary2.TryGetValue(item2.manifest.id, out value5) ? value5 : 999999999);
					priorityQueue.Enqueue(item2, priority2);
				}
			}
		}
		bool flag = false;
		if (_mods.Count != list4.Count)
		{
			Log.Error($"We found {_mods.Count} mods, but after sorting, we only have {list4.Count}! This should never happen");
		}
		if (manualOrdering.Count != list4.Count)
		{
			flag = true;
		}
		else
		{
			for (int num6 = 0; num6 < manualOrdering.Count; num6++)
			{
				if (manualOrdering[num6].Id != list4[num6].manifest?.id)
				{
					flag = true;
					break;
				}
			}
		}
		if (flag)
		{
			Log.Info("Mods have been re-sorted because we detected a change or dependency order was broken. New sorting order:");
			for (int num7 = 0; num7 < list4.Count; num7++)
			{
				Log.Info($"  {num7}: {list4[num7].manifest?.name} ({list4[num7].manifest?.id})");
			}
		}
		_mods = list4;
	}

	private static List<int>? HasCircularDependenciesRecursive(int modIndex, HashSet<int> seenMods, List<int> currentChain, List<int> dependencyCounts)
	{
		if (currentChain.Contains(modIndex))
		{
			List<int> list = currentChain.ToList();
			list.Add(modIndex);
			dependencyCounts[modIndex]--;
			return list;
		}
		if (seenMods.Contains(modIndex))
		{
			return null;
		}
		seenMods.Add(modIndex);
		Mod mod = _mods[modIndex];
		currentChain.Add(modIndex);
		List<int> list2 = null;
		if (mod.manifest?.dependencies != null)
		{
			foreach (string dependencyName in mod.manifest.dependencies)
			{
				int num = _mods.FindIndex((Mod m) => m.manifest?.id == dependencyName);
				if (num >= 0)
				{
					List<int> list3 = HasCircularDependenciesRecursive(num, seenMods, currentChain, dependencyCounts);
					if (list2 == null)
					{
						list2 = list3;
					}
				}
			}
		}
		currentChain.RemoveAt(currentChain.Count - 1);
		return list2;
	}

	private static void ReadModsInDirRecursive(string path, ModSource source, List<Mod>? newMods)
	{
		string[] array = _fileIo?.GetFilesAt(path) ?? Array.Empty<string>();
		foreach (string text in array)
		{
			if (text.EndsWith(".json"))
			{
				string text2 = Path.Combine(path, text);
				Log.Info("Found mod manifest file " + text2);
				Mod mod = ReadModManifest(text2, source);
				if (mod != null)
				{
					_mods.Add(mod);
					newMods?.Add(mod);
				}
			}
		}
		string[] array2 = _fileIo?.GetDirectoriesAt(path) ?? Array.Empty<string>();
		foreach (string path2 in array2)
		{
			string path3 = Path.Combine(path, path2);
			if (_fileIo.DirectoryExists(path3))
			{
				ReadModsInDirRecursive(path3, source, newMods);
			}
		}
	}

	private static Mod? ReadModManifest(string filename, ModSource source)
	{
		if (_fileIo == null)
		{
			return null;
		}
		try
		{
			using Stream utf8Json = _fileIo.OpenStream(filename, Godot.FileAccess.ModeFlags.Read);
			ModManifest modManifest = JsonSerializer.Deserialize(utf8Json, JsonSerializationUtility.GetTypeInfo<ModManifest>());
			if (modManifest == null)
			{
				throw new InvalidOperationException("JSON deserialization returned null when trying to deserialize mod manifest!");
			}
			if (modManifest.id == null)
			{
				Log.Error("Mod manifest " + filename + " is missing the 'id' field! This is not allowed. The mod will not be loaded.");
				return null;
			}
			return new Mod
			{
				path = filename.GetBaseDir(),
				modSource = source,
				manifest = modManifest
			};
		}
		catch (Exception ex)
		{
			Log.Error($"Caught {ex.GetType()} trying to deserialize mod manifest json at path {filename}:\n{ex}");
			return null;
		}
	}

	private static void ReadSteamMods()
	{
		uint numSubscribedItems = SteamUGC.GetNumSubscribedItems();
		PublishedFileId_t[] array = new PublishedFileId_t[numSubscribedItems];
		numSubscribedItems = SteamUGC.GetSubscribedItems(array, numSubscribedItems);
		for (int i = 0; i < numSubscribedItems; i++)
		{
			PublishedFileId_t workshopItemId = array[i];
			TryReadModFromSteam(workshopItemId, null);
		}
		_steamItemInstalledCallback = Callback<ItemInstalled_t>.Create(OnSteamWorkshopItemInstalled);
	}

	private static void TryReadModFromSteam(PublishedFileId_t workshopItemId, List<Mod>? newMods)
	{
		if (!SteamUGC.GetItemInstallInfo(workshopItemId, out var punSizeOnDisk, out var pchFolder, 256u, out var punTimeStamp))
		{
			Log.Warn($"Could not get Steam Workshop item install info for item {workshopItemId.m_PublishedFileId}");
			return;
		}
		Log.Info($"Looking for mods to load from Steam Workshop mod {workshopItemId.m_PublishedFileId} in {pchFolder} (size {punSizeOnDisk}, last modified {punTimeStamp})");
		if (_fileIo != null && !_fileIo.DirectoryExists(pchFolder))
		{
			Log.Warn("Could not open Steam Workshop folder: " + pchFolder);
		}
		else
		{
			ReadModsInDirRecursive(pchFolder, ModSource.SteamWorkshop, newMods);
		}
	}

	private static void OnSteamWorkshopItemInstalled(ItemInstalled_t ev)
	{
		if ((ulong)ev.m_unAppID.m_AppId != 2868840)
		{
			return;
		}
		Log.Info($"Detected new Steam Workshop item installation, id: {ev.m_nPublishedFileId.m_PublishedFileId}");
		List<Mod> list = new List<Mod>();
		TryReadModFromSteam(ev.m_nPublishedFileId, list);
		foreach (Mod item in list)
		{
			item.state = ModLoadState.AddedAtRuntime;
			ModManager.OnModDetected?.Invoke(item);
		}
	}

	private static void TryLoadMod(Mod mod)
	{
		Assembly assembly = null;
		if (mod.manifest == null)
		{
			throw new InvalidOperationException("Tried to load mod before its manifest was loaded!");
		}
		string modId = mod.manifest.id;
		bool flag = _settings?.IsModDisabled(modId, mod.modSource) ?? false;
		bool flag2 = _mods.Any((Mod m) => m.manifest?.id == modId && m.state == ModLoadState.Loaded);
		string value;
		if (_initialized)
		{
			Log.Info("Skipping loading mod " + modId + ", can't load mods at runtime");
			mod.state = ModLoadState.AddedAtRuntime;
		}
		else if (flag)
		{
			Log.Info("Skipping loading mod " + modId + ", it is set to disabled in settings");
			mod.state = ModLoadState.Disabled;
		}
		else if (!PlayerAgreedToModLoading)
		{
			Log.Info("Skipping loading mod " + modId + ", user has not yet seen the mods warning");
			mod.state = ModLoadState.Disabled;
		}
		else if (flag2)
		{
			LocString locString = new LocString("main_menu_ui", "MOD_ERROR.DUPLICATE_ID");
			locString.Add("id", modId);
			Log.Error("Tried to load mod with id " + modId + ", but a mod is already loaded with that name!");
			mod.state = ModLoadState.Failed;
			int num = 1;
			List<LocString> list = new List<LocString>(num);
			CollectionsMarshal.SetCount(list, num);
			Span<LocString> span = CollectionsMarshal.AsSpan(list);
			int index = 0;
			span[index] = locString;
			mod.errors = list;
		}
		else if (_circularDependencies.TryGetValue(modId, out value))
		{
			LocString locString2 = new LocString("main_menu_ui", "MOD_ERROR.CIRCULAR_DEPENDENCY");
			locString2.Add("id", modId);
			locString2.Add("dependencyChain", value);
			Log.Error($"Tried to load mod with id {modId}, but it is part of a circular dependency chain: {value}!");
			mod.state = ModLoadState.Failed;
			int index = 1;
			List<LocString> list2 = new List<LocString>(index);
			CollectionsMarshal.SetCount(list2, index);
			Span<LocString> span = CollectionsMarshal.AsSpan(list2);
			int num = 0;
			span[num] = locString2;
			mod.errors = list2;
		}
		else
		{
			HashSet<string> hashSet = new HashSet<string>();
			List<string>? dependencies = mod.manifest.dependencies;
			if (dependencies != null && dependencies.Count > 0)
			{
				foreach (Mod mod2 in _mods)
				{
					if (mod2.state == ModLoadState.Loaded && mod2.manifest?.id != null && mod.manifest.dependencies.Contains(mod2.manifest.id))
					{
						hashSet.Add(mod2.manifest.id);
					}
				}
				if (hashSet.Count != mod.manifest.dependencies.Count)
				{
					List<string> list3 = new List<string>();
					foreach (string dependency in mod.manifest.dependencies)
					{
						if (!hashSet.Contains(dependency))
						{
							list3.Add(dependency);
						}
					}
					string text = string.Join(",", list3);
					LocString locString3 = new LocString("main_menu_ui", "MOD_ERROR.MISSING_DEPENDENCY");
					locString3.Add("id", mod.manifest.id);
					locString3.Add("missingCount", list3.Count);
					locString3.Add("missingDependencies", text);
					Log.Error($"Tried to load mod {modId}, but it depends on mods which have not been loaded: {text}!");
					mod.state = ModLoadState.Failed;
					int num = 1;
					List<LocString> list4 = new List<LocString>(num);
					CollectionsMarshal.SetCount(list4, num);
					Span<LocString> span = CollectionsMarshal.AsSpan(list4);
					int index = 0;
					span[index] = locString3;
					mod.errors = list4;
				}
			}
		}
		if (mod.state != ModLoadState.None)
		{
			ModManager.OnModDetected?.Invoke(mod);
			return;
		}
		List<LocString> list5 = null;
		try
		{
			bool flag3 = false;
			string text2 = Path.Combine(mod.path, modId + ".dll");
			if (mod.manifest.hasDll)
			{
				if (_fileIo != null && _fileIo.FileExists(text2))
				{
					Log.Info("Loading assembly DLL " + text2);
					AssemblyLoadContext loadContext = AssemblyLoadContext.GetLoadContext(Assembly.GetExecutingAssembly());
					if (loadContext != null)
					{
						assembly = loadContext.LoadFromAssemblyPath(text2);
						flag3 = true;
					}
				}
				else
				{
					Log.Error($"Mod manifest for mod {mod.manifest.id} declares that it should load an assembly, but no assembly at path {text2} was found!");
				}
			}
			string text3 = Path.Combine(mod.path, modId + ".pck");
			if (mod.manifest.hasPck)
			{
				if (_fileIo != null && _fileIo.FileExists(text3))
				{
					Log.Info("Loading Godot PCK " + text3);
					if (!ProjectSettings.LoadResourcePack(text3))
					{
						throw new InvalidOperationException("Godot errored while loading PCK file " + modId + "!");
					}
					flag3 = true;
				}
				else
				{
					Log.Error($"Mod manifest for mod {mod.manifest.id} declares that it should load a PCK, but no PCK at path {text3} was found!");
				}
			}
			if (!flag3)
			{
				Log.Warn("Neither a DLL nor a PCK was loaded for mod " + mod.manifest.id + ", something seems wrong!");
			}
			bool? flag4 = null;
			if (assembly != null)
			{
				flag4 = true;
				List<Type> list6 = (from t in assembly.GetTypes()
					where t.GetCustomAttribute<ModInitializerAttribute>() != null
					select t).ToList();
				if (list6.Count > 0)
				{
					foreach (Type item in list6)
					{
						Log.Info($"Calling initializer method of type {item} for {assembly}");
						bool flag5 = CallModInitializer(item);
						flag4 = flag4.Value && flag5;
					}
				}
				else
				{
					try
					{
						Log.Info($"No ModInitializerAttribute detected. Calling Harmony.PatchAll for {assembly}");
						Harmony harmony = new Harmony((mod.manifest.author ?? "unknown") + "." + modId);
						harmony.PatchAll(assembly);
					}
					catch (Exception value2)
					{
						Log.Error($"Exception caught while trying to run PatchAll on assembly {assembly}:\n{value2}");
						flag4 = false;
					}
				}
			}
			if (flag4 == false)
			{
				if (list5 == null)
				{
					list5 = new List<LocString>();
				}
				LocString locString4 = new LocString("main_menu_ui", "MOD_ERROR.ASSEMBLY_LOAD");
				locString4.Add("id", mod.manifest.id);
				list5.Add(locString4);
			}
			Log.Info($"Finished mod initialization for '{mod.manifest.name}' ({modId}).");
			mod.state = ModLoadState.Loaded;
			mod.assembly = assembly;
			mod.errors = list5;
			ModManager.OnModDetected?.Invoke(mod);
		}
		catch (Exception ex)
		{
			Log.Error($"Exception thrown while loading mod {modId}: {ex}");
			if (list5 == null)
			{
				list5 = new List<LocString>();
			}
			LocString locString5 = new LocString("main_menu_ui", "MOD_ERROR.EXCEPTION");
			locString5.Add("exceptionType", ex.GetType().ToString());
			locString5.Add("id", mod.manifest.id);
			list5.Add(locString5);
			mod.state = ModLoadState.Failed;
			mod.assembly = assembly;
			mod.errors = list5;
			ModManager.OnModDetected?.Invoke(mod);
		}
	}

	private static bool CallModInitializer(Type initializerType)
	{
		ModInitializerAttribute customAttribute = initializerType.GetCustomAttribute<ModInitializerAttribute>();
		MethodInfo method = initializerType.GetMethod(customAttribute.initializerMethod, BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);
		if (method == null)
		{
			method = initializerType.GetMethod(customAttribute.initializerMethod, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
			if (method != null)
			{
				Log.Error($"Tried to call mod initializer {initializerType.Name}.{customAttribute.initializerMethod} but it's not static! Declare it to be static");
			}
			else
			{
				Log.Error($"Found mod initializer class of type {initializerType}, but it does not contain the method {customAttribute.initializerMethod} declared in the ModInitializerAttribute!");
			}
			return false;
		}
		try
		{
			method.Invoke(null, null);
		}
		catch (Exception value)
		{
			Log.Error($"Exception thrown when calling mod initializer of type {initializerType}: {value}");
			return false;
		}
		return true;
	}

	public static IEnumerable<string> GetModdedLocTables(string language, string file)
	{
		foreach (Mod mod in _mods)
		{
			if (mod.state == ModLoadState.Loaded)
			{
				string text = $"res://{mod.manifest.id}/localization/{language}/{file}";
				if (ResourceLoader.Exists(text))
				{
					yield return text;
				}
			}
		}
	}

	public static List<string>? GetGameplayRelevantModNameList()
	{
		if (!IsRunningModded())
		{
			return null;
		}
		return (from m in GetLoadedMods()
			where m.manifest?.affectsGameplay ?? true
			select m.manifest?.id + "-" + m.manifest?.version).ToList();
	}

	private static Assembly HandleAssemblyResolveFailure(object? source, ResolveEventArgs ev)
	{
		if (ev.Name.StartsWith("sts2,"))
		{
			Log.Info($"Failed to resolve assembly '{ev.Name}' but it looks like the STS2 assembly. Resolving using {Assembly.GetExecutingAssembly()}");
			return Assembly.GetExecutingAssembly();
		}
		if (ev.Name.StartsWith("0Harmony,"))
		{
			Log.Info($"Failed to resolve assembly '{ev.Name}' but it looks like the Harmony assembly. Resolving using {typeof(Harmony).Assembly}");
			return typeof(Harmony).Assembly;
		}
		return null;
	}

	public static void CallMetricsHooks(SerializableRun run, bool isVictory, ulong localPlayerId)
	{
		ModManager.OnMetricsUpload?.Invoke(run, isVictory, localPlayerId);
	}

	public static bool IsRunningModded()
	{
		return _mods.Any(delegate(Mod m)
		{
			ModLoadState state = m.state;
			return (uint)(state - 1) <= 1u;
		});
	}

	public static bool HasHarmonyPatches()
	{
		try
		{
			bool valueOrDefault = _hasHarmonyPatches == true;
			if (!_hasHarmonyPatches.HasValue)
			{
				valueOrDefault = Harmony.GetAllPatchedMethods().Any();
				_hasHarmonyPatches = valueOrDefault;
			}
		}
		catch
		{
			_hasHarmonyPatches = true;
		}
		return _hasHarmonyPatches.Value;
	}

	public static IEnumerable<Mod> GetLoadedMods()
	{
		return _mods.Where((Mod m) => m.state == ModLoadState.Loaded);
	}

	public static void Dispose()
	{
		_steamItemInstalledCallback?.Dispose();
	}
}
