using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.Loader;
using System.Text.Json;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Platform.Steam;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;
using Steamworks;

namespace MegaCrit.Sts2.Core.Modding;

public static class ModManager
{
	public delegate void MetricsUploadHook(SerializableRun run, bool isVictory, ulong localPlayerId);

	private static bool _allowInitForTests = false;

	private static List<Mod> _mods = new List<Mod>();

	private static List<Mod> _loadedMods = new List<Mod>();

	private static bool _initialized;

	private static Callback<ItemInstalled_t>? _steamItemInstalledCallback;

	private static ModSettings? _settings;

	private static IModManagerFileIo? _fileIo;

	public static IReadOnlyList<Mod> AllMods => _mods;

	public static IReadOnlyList<Mod> LoadedMods => _loadedMods;

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
			_loadedMods = _mods.Where((Mod m) => m.wasLoaded).ToList();
			if (_loadedMods.Count > 0)
			{
				Log.Info($" --- RUNNING MODDED! --- Loaded {_loadedMods.Count} mods ({_mods.Count} total)");
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
		_loadedMods.Clear();
		_initialized = false;
		_settings = null;
		_fileIo = null;
		_allowInitForTests = true;
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
					if (mod2 == null)
					{
						Log.Error($"Mod {mod.manifest.name} lists dependency {dependencyName} which was not found! Mod may not function correctly");
					}
					else
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
			BreakCircularDependenciesRecursive(num2, seenMods, currentChain, list);
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
				int value2;
				int priority = (dictionary2.TryGetValue(mod3.manifest.id, out value2) ? value2 : 999999999);
				priorityQueue.Enqueue(mod3, priority);
			}
		}
		if (priorityQueue.Count == 0)
		{
			Log.Error($"Detected {_mods.Count} mods, but all of them have dependencies! Something seems wrong");
		}
		List<Mod> list3 = new List<Mod>();
		while (priorityQueue.Count > 0)
		{
			Mod mod4 = priorityQueue.Dequeue();
			list3.Add(mod4);
			if (!dictionary.TryGetValue(mod4, out var value3))
			{
				continue;
			}
			foreach (Mod item in value3)
			{
				int num5 = _mods.IndexOf(item);
				if (num5 < 0)
				{
					throw new InvalidOperationException("Bug in mod sorting logic!");
				}
				list[num5]--;
				if (list[num5] == 0)
				{
					int value4;
					int priority2 = (dictionary2.TryGetValue(item.manifest.id, out value4) ? value4 : 999999999);
					priorityQueue.Enqueue(item, priority2);
				}
			}
		}
		bool flag = false;
		if (_mods.Count != list3.Count)
		{
			Log.Error($"We found {_mods.Count} mods, but after sorting, we only have {list3.Count}! This should never happen");
		}
		if (manualOrdering.Count != list3.Count)
		{
			flag = true;
		}
		else
		{
			for (int num6 = 0; num6 < manualOrdering.Count; num6++)
			{
				if (manualOrdering[num6].Id != list3[num6].manifest?.id)
				{
					flag = true;
					break;
				}
			}
		}
		if (flag)
		{
			Log.Info("Mods have been re-sorted because we detected a change or dependency order was broken. New sorting order:");
			for (int num7 = 0; num7 < list3.Count; num7++)
			{
				Log.Info($"  {num7}: {list3[num7].manifest?.name} ({list3[num7].manifest?.id})");
			}
		}
		_mods = list3;
	}

	private static void BreakCircularDependenciesRecursive(int modIndex, HashSet<int> seenMods, List<int> currentChain, List<int> inDegrees)
	{
		if (currentChain.Contains(modIndex))
		{
			string text = string.Join(",", currentChain.Select((int i) => _mods[i].manifest?.id));
			text += ",";
			text += _mods[modIndex].manifest?.id;
			Log.Error("Detected circular dependency chain: " + text + ". Breaking last dependency!");
			inDegrees[modIndex]--;
		}
		else
		{
			if (seenMods.Contains(modIndex))
			{
				return;
			}
			seenMods.Add(modIndex);
			Mod mod = _mods[modIndex];
			currentChain.Add(modIndex);
			if (mod.manifest?.dependencies != null)
			{
				foreach (string dependencyName in mod.manifest.dependencies)
				{
					int num = _mods.FindIndex((Mod m) => m.manifest?.id == dependencyName);
					if (num >= 0)
					{
						BreakCircularDependenciesRecursive(num, seenMods, currentChain, inDegrees);
					}
				}
			}
			currentChain.RemoveAt(currentChain.Count - 1);
		}
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
		bool flag2 = _mods.Any((Mod m) => m.manifest?.id == modId && m.wasLoaded);
		if (!PlayerAgreedToModLoading || flag || flag2 || _initialized)
		{
			if (_initialized)
			{
				Log.Info("Skipping loading mod " + modId + ", can't load mods at runtime");
			}
			else if (flag)
			{
				Log.Info("Skipping loading mod " + modId + ", it is set to disabled in settings");
			}
			else if (!PlayerAgreedToModLoading)
			{
				Log.Info("Skipping loading mod " + modId + ", user has not yet seen the mods warning");
			}
			else if (flag2)
			{
				Log.Warn("Tried to load mod with id " + modId + ", but a mod is already loaded with that name!");
			}
			mod.wasLoaded = false;
			ModManager.OnModDetected?.Invoke(mod);
			return;
		}
		try
		{
			bool flag3 = false;
			string text = Path.Combine(mod.path, modId + ".dll");
			if (mod.manifest.hasDll)
			{
				if (_fileIo != null && _fileIo.FileExists(text))
				{
					Log.Info("Loading assembly DLL " + text);
					AssemblyLoadContext loadContext = AssemblyLoadContext.GetLoadContext(Assembly.GetExecutingAssembly());
					if (loadContext != null)
					{
						assembly = loadContext.LoadFromAssemblyPath(text);
						flag3 = true;
					}
				}
				else
				{
					Log.Error($"Mod manifest for mod {mod.manifest.id} declares that it should load an assembly, but no assembly at path {text} was found!");
				}
			}
			string text2 = Path.Combine(mod.path, modId + ".pck");
			if (mod.manifest.hasPck)
			{
				if (_fileIo != null && _fileIo.FileExists(text2))
				{
					Log.Info("Loading Godot PCK " + text2);
					if (!ProjectSettings.LoadResourcePack(text2))
					{
						throw new InvalidOperationException("Godot errored while loading PCK file " + modId + "!");
					}
					flag3 = true;
				}
				else
				{
					Log.Error($"Mod manifest for mod {mod.manifest.id} declares that it should load a PCK, but no PCK at path {text2} was found!");
				}
			}
			if (!flag3)
			{
				Log.Warn("Neither a DLL nor a PCK was loaded for mod " + mod.manifest.id + ", something seems wrong!");
			}
			bool? assemblyLoadedSuccessfully = null;
			if (assembly != null)
			{
				assemblyLoadedSuccessfully = true;
				List<Type> list = (from t in assembly.GetTypes()
					where t.GetCustomAttribute<ModInitializerAttribute>() != null
					select t).ToList();
				if (list.Count > 0)
				{
					foreach (Type item in list)
					{
						Log.Info($"Calling initializer method of type {item} for {assembly}");
						bool flag4 = CallModInitializer(item);
						assemblyLoadedSuccessfully = assemblyLoadedSuccessfully.Value && flag4;
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
					catch (Exception value)
					{
						Log.Error($"Exception caught while trying to run PatchAll on assembly {assembly}:\n{value}");
						assemblyLoadedSuccessfully = false;
					}
				}
			}
			Log.Info($"Finished mod initialization for '{mod.manifest.name}' ({modId}).");
			mod.wasLoaded = true;
			mod.assembly = assembly;
			mod.assemblyLoadedSuccessfully = assemblyLoadedSuccessfully;
			ModManager.OnModDetected?.Invoke(mod);
		}
		catch (Exception value2)
		{
			Log.Error($"Error loading mod {modId}: {value2}");
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
			if (mod.wasLoaded)
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
		if (LoadedMods.Count == 0)
		{
			return null;
		}
		return (from m in LoadedMods
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

	public static void Dispose()
	{
		_steamItemInstalledCallback?.Dispose();
	}
}
