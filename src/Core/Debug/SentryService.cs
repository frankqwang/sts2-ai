using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Platform.Steam;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using Sentry;

namespace MegaCrit.Sts2.Core.Debug;

public static class SentryService
{
	private const string _dsnSettingPath = "sentry/config/dsn";

	private static readonly StringName _sentrySdkSingleton = new StringName("SentrySDK");

	private static readonly StringName _sentryUserClass = new StringName("SentryUser");

	private static readonly StringName _sentryBreadcrumbClass = new StringName("SentryBreadcrumb");

	private static readonly StringName _levelProperty = new StringName("level");

	private static readonly StringName _categoryProperty = new StringName("category");

	private static readonly StringName _idProperty = new StringName("id");

	private static readonly StringName _createMethod = new StringName("create");

	private static readonly StringName _setUserMethod = new StringName("set_user");

	private static readonly StringName _addBreadcrumbMethod = new StringName("add_breadcrumb");

	private static readonly StringName _shutdownMethod = new StringName("shutdown");

	private static readonly StringName _setShouldSampleEventMethod = new StringName("set_should_sample_event");

	private static readonly StringName _setPlatformBranchMethod = new StringName("set_platform_branch");

	private static readonly StringName _setCsharpContextMethod = new StringName("set_csharp_context");

	private static IDisposable? _sentryInstance;

	private static float _sampleRate = 1f;

	private static readonly string _sessionId = Guid.NewGuid().ToString();

	private static Node? _sentryInit;

	private static GodotObject? _extensionSdk;

	public static bool IsEnabled { get; private set; }

	public static bool SampleForNonSteamBranches { get; private set; }

	public static bool IsForcedOn { get; private set; }

	public static string SessionId => _sessionId;

	public static void Initialize()
	{
		bool flag = OS.HasFeature("editor");
		bool flag2 = DisplayServer.GetName().Equals("headless", StringComparison.OrdinalIgnoreCase);
		bool isForcedOn = CommandLineHelper.HasArg("force-sentry");
		if (flag && !flag2 && !isForcedOn)
		{
			Log.Info("[Sentry.NET] Disabled in editor");
			return;
		}
		SampleForNonSteamBranches = flag2 || isForcedOn;
		IsForcedOn = isForcedOn;
		string dsn = GetDsn();
		if (string.IsNullOrEmpty(dsn))
		{
			Log.Info("[Sentry.NET] Disabled: no DSN configured in project settings");
			return;
		}
		ReleaseInfo releaseInfo = ReleaseInfoManager.Instance.ReleaseInfo;
		string environment = "development";
		string release = releaseInfo?.Version ?? "dev";
		_sentryInstance = SentrySdk.Init(delegate(SentryOptions options)
		{
			options.Dsn = dsn;
			options.Environment = environment;
			options.Release = release;
			options.Debug = isForcedOn;
			options.AutoSessionTracking = true;
			options.IsGlobalModeEnabled = true;
			options.SendDefaultPii = false;
			options.SetBeforeSend(delegate(SentryEvent sentryEvent, SentryHint hint)
			{
				if (sentryEvent.Exception is AutoSlayTimeoutException)
				{
					return (SentryEvent?)null;
				}
				return (!ShouldSampleEvent()) ? null : sentryEvent;
			});
		});
		IsEnabled = SentrySdk.IsEnabled;
		if (!IsEnabled)
		{
			Log.Warn("[Sentry.NET] SDK initialization failed");
			return;
		}
		SentrySdk.ConfigureScope(delegate(Scope scope)
		{
			scope.SetTag("sdk", "dotnet");
			scope.SetTag("session_id", _sessionId);
			scope.SetExtra("assembly.main_hash", AssemblyHasher.GetMainAssemblyHash());
			if (releaseInfo != null)
			{
				scope.SetTag("branch", releaseInfo.Branch);
				scope.SetExtra("build.commit", releaseInfo.Commit);
				scope.SetExtra("build.main_hash", releaseInfo.MainAssemblyHash);
				scope.SetExtra("build.date", releaseInfo.Date.ToString("o"));
			}
		});
		Log.LogCallback += OnLogCallback;
		Log.Info("[Sentry.NET] Initialized: env=" + environment + ", release=" + release);
	}

	public static void AfterGameInit(string? platformBranch, string uniqueId, Node treeRoot)
	{
		if (!IsEnabled)
		{
			return;
		}
		_sentryInit = treeRoot.GetNode("SentryInit");
		_sentryInit?.Call(_setCsharpContextMethod, _sessionId, AssemblyHasher.GetMainAssemblyHash());
		if (!ShouldStayAliveAfterInit())
		{
			Log.Info("[Sentry.NET] Shutting down because event reporting is disabled.");
			Shutdown();
			return;
		}
		SentrySdk.ConfigureScope(delegate(Scope scope)
		{
			scope.User = new SentryUser
			{
				Id = uniqueId
			};
		});
		SetGdExtensionUser(uniqueId);
		Log.Debug("[Sentry.NET] User context set");
		SetPlatformBranch(platformBranch);
	}

	private static void OnLogCallback(LogLevel level, string message, int skipFrames)
	{
		if (IsEnabled)
		{
			switch (level)
			{
			case LogLevel.Error:
				SentrySdk.AddBreadcrumb(message, "log", null, null, BreadcrumbLevel.Error);
				SetGdExtensionBreadcrumb(message, "log", BreadcrumbLevel.Error);
				break;
			case LogLevel.Warn:
				SentrySdk.AddBreadcrumb(message, "log", null, null, BreadcrumbLevel.Warning);
				SetGdExtensionBreadcrumb(message, "log", BreadcrumbLevel.Warning);
				break;
			}
		}
	}

	private static void SetGdExtensionUser(string uniqueId)
	{
		try
		{
			if (Engine.HasSingleton(_sentrySdkSingleton))
			{
				if (_extensionSdk == null)
				{
					_extensionSdk = Engine.GetSingleton(_sentrySdkSingleton);
				}
				GodotObject godotObject = ClassDB.Instantiate(_sentryUserClass).AsGodotObject();
				godotObject.Set(_idProperty, uniqueId);
				_extensionSdk.Call(_setUserMethod, godotObject);
			}
		}
		catch (Exception ex)
		{
			Log.Warn("[Sentry] Failed to set GDExtension user: " + ex.Message);
		}
	}

	private static void SetGdExtensionBreadcrumb(string message, string category, BreadcrumbLevel level)
	{
		try
		{
			if (Engine.HasSingleton(_sentrySdkSingleton))
			{
				if (_extensionSdk == null)
				{
					_extensionSdk = Engine.GetSingleton(_sentrySdkSingleton);
				}
				GodotObject godotObject = ClassDB.ClassCallStatic(_sentryBreadcrumbClass, _createMethod, message).AsGodotObject();
				godotObject.Set(_categoryProperty, category);
				godotObject.Set(_levelProperty, (int)(level + 1));
				_extensionSdk.Call(_addBreadcrumbMethod, godotObject);
			}
		}
		catch (Exception ex)
		{
			Log.Warn("[Sentry] Failed to set GDExtension breadcrumb: " + ex.Message);
		}
	}

	private static void SetPlatformBranch(string? branch)
	{
		_sampleRate = branch switch
		{
			"public" => 0.1f, 
			"private-beta" => 1f, 
			"public-beta" => 0.2f, 
			_ => (branch != null) ? 0.1f : (SampleForNonSteamBranches ? 1f : 0f), 
		};
		_sentryInit?.Call(_setShouldSampleEventMethod, Callable.From((Func<bool>)ShouldSampleEvent));
		if (branch != null)
		{
			_sentryInit?.Call(_setPlatformBranchMethod, branch);
		}
		if (IsEnabled)
		{
			if (_sampleRate == 0f)
			{
				Log.Info("[Sentry.NET] Disabled: no platform branch (non-Steam build)");
				Shutdown();
				return;
			}
			if (branch != null)
			{
				SentrySdk.ConfigureScope(delegate(Scope scope)
				{
					scope.SetTag("platform.branch", branch);
					scope.Environment = branch;
				});
			}
		}
		Log.Info($"[Sentry.NET] Platform branch: {branch}, sample rate: {_sampleRate:P0}");
	}

	public static void AddBreadcrumb(string message, string category = "app", BreadcrumbLevel level = BreadcrumbLevel.Info)
	{
		if (IsEnabled)
		{
			SentrySdk.AddBreadcrumb(message, category, null, null, level);
		}
	}

	public static void CaptureException(Exception ex)
	{
		if (IsEnabled)
		{
			SentrySdk.CaptureException(ex, delegate(Scope scope)
			{
				AttachGameState(scope);
			});
		}
	}

	public static void CaptureException(Exception ex, Action<Scope> configureScope)
	{
		if (IsEnabled)
		{
			SentrySdk.CaptureException(ex, delegate(Scope scope)
			{
				AttachGameState(scope);
				configureScope(scope);
			});
		}
	}

	public static void CaptureMessage(string message, SentryLevel level = SentryLevel.Info, Action<Scope>? configureScope = null)
	{
		if (IsEnabled)
		{
			SentryEvent evt = new SentryEvent
			{
				Message = message,
				Level = level
			};
			SentrySdk.CaptureEvent(evt, delegate(Scope scope)
			{
				AttachGameState(scope);
				configureScope?.Invoke(scope);
			});
		}
	}

	public static void SetTag(string key, string value)
	{
		if (IsEnabled)
		{
			SentrySdk.ConfigureScope(delegate(Scope scope)
			{
				scope.SetTag(key, value);
			});
		}
	}

	public static void SetExtra(string key, object value)
	{
		if (IsEnabled)
		{
			SentrySdk.ConfigureScope(delegate(Scope scope)
			{
				scope.SetExtra(key, value);
			});
		}
	}

	public static void InitializeForTesting()
	{
		if (IsEnabled)
		{
			return;
		}
		string dsn = GetDsn();
		if (string.IsNullOrEmpty(dsn))
		{
			Log.Warn("[Sentry.NET] Cannot initialize for testing: no DSN configured");
			return;
		}
		_sentryInstance = SentrySdk.Init(delegate(SentryOptions options)
		{
			options.Dsn = dsn;
			options.Environment = "development";
			options.Release = ReleaseInfoManager.Instance.ReleaseInfo?.Version ?? "dev-console-test";
			options.Debug = false;
			options.AutoSessionTracking = false;
			options.IsGlobalModeEnabled = true;
			options.SendDefaultPii = false;
		});
		IsEnabled = SentrySdk.IsEnabled;
		if (!IsEnabled)
		{
			Log.Warn("[Sentry.NET] SDK initialization failed for testing");
			return;
		}
		SentrySdk.ConfigureScope(delegate(Scope scope)
		{
			scope.SetTag("sdk", "dotnet");
			scope.SetTag("session_id", _sessionId);
			scope.SetTag("source", "dev-console-test");
		});
		Log.Info("[Sentry.NET] Initialized for testing via dev console");
	}

	public static void Shutdown()
	{
		if (IsEnabled)
		{
			Log.LogCallback -= OnLogCallback;
			Log.Info("[Sentry.NET] Shutting down");
			_sentryInstance?.Dispose();
			_sentryInstance = null;
			_sentryInit?.Call(_shutdownMethod);
			IsEnabled = false;
		}
	}

	private static string GetDsn()
	{
		return ProjectSettings.GetSetting("sentry/config/dsn", "").AsString();
	}

	private static void AttachGameState(Scope scope)
	{
		try
		{
			scope.SetExtra("loc.language", LocManager.Instance.Language);
			string currentSceneName = GetCurrentSceneName();
			if (currentSceneName != null)
			{
				scope.SetTag("game.scene", currentSceneName);
			}
			RunState runState = RunManager.Instance.DebugOnlyGetState();
			if (RunManager.Instance.IsInProgress && runState != null)
			{
				scope.SetTag("game.in_run", "true");
				scope.SetExtra("game.seed", runState.Rng.StringSeed);
				scope.SetExtra("game.ascension", runState.AscensionLevel);
				scope.SetExtra("game.act", runState.CurrentActIndex + 1);
				scope.SetExtra("game.act_name", runState.Act.Id.ToString());
				scope.SetExtra("game.floor", runState.TotalFloor);
				scope.SetExtra("game.mode", runState.GameMode);
				AbstractRoom currentRoom = runState.CurrentRoom;
				scope.SetExtra("game.room_type", currentRoom?.GetType().Name);
				if (currentRoom is EventRoom eventRoom)
				{
					scope.SetExtra("game.event", eventRoom.CanonicalEvent.Id.Entry);
				}
				IReadOnlyList<Player> players = runState.Players;
				if (players.Count > 0)
				{
					scope.SetExtra("game.characters", string.Join(", ", players.Select((Player p) => p.Character.Id)));
					scope.SetExtra("game.player_count", players.Count);
				}
			}
			else
			{
				scope.SetTag("game.in_run", "false");
			}
			CombatState combatState = CombatManager.Instance.DebugOnlyGetState();
			if (combatState != null)
			{
				scope.SetExtra("combat.encounter", combatState.Encounter?.Id.Entry);
				scope.SetExtra("combat.round", combatState.RoundNumber);
				scope.SetExtra("combat.enemy_count", combatState.Enemies.Count);
				scope.SetExtra("combat.enemies", string.Join(", ", combatState.Enemies.Select((Creature e) => e.Monster?.Id.ToString() ?? "unknown")));
				List<string> list = combatState.Players.Select((Player p) => $"{p.Creature.CurrentHp}/{p.Creature.MaxHp}").ToList();
				if (list.Count > 0)
				{
					scope.SetExtra("combat.player_hp", string.Join(", ", list));
				}
			}
		}
		catch
		{
		}
	}

	private static string? GetCurrentSceneName()
	{
		try
		{
			NGame instance = NGame.Instance;
			if (instance == null)
			{
				return null;
			}
			if (instance.MainMenu != null)
			{
				return "MainMenu";
			}
			if (instance.CurrentRunNode != null)
			{
				NRun currentRunNode = instance.CurrentRunNode;
				if (currentRunNode.CombatRoom != null)
				{
					return "CombatRoom";
				}
				if (currentRunNode.MapRoom != null)
				{
					return "MapRoom";
				}
				if (currentRunNode.EventRoom != null)
				{
					return "EventRoom";
				}
				if (currentRunNode.RestSiteRoom != null)
				{
					return "RestSiteRoom";
				}
				if (currentRunNode.MerchantRoom != null)
				{
					return "MerchantRoom";
				}
				if (currentRunNode.TreasureRoom != null)
				{
					return "TreasureRoom";
				}
				return "Run";
			}
			if (instance.LogoAnimation != null)
			{
				return "LogoAnimation";
			}
			return null;
		}
		catch
		{
			return null;
		}
	}

	private static bool ShouldSampleEvent()
	{
		if (System.Random.Shared.NextDouble() >= (double)_sampleRate)
		{
			return false;
		}
		if (!SaveManager.Instance.PrefsSave.UploadData)
		{
			return false;
		}
		return true;
	}

	private static bool ShouldStayAliveAfterInit()
	{
		if (IsForcedOn)
		{
			Log.Info("[Sentry.NET] Staying alive because we're forced on");
			return true;
		}
		if (!SteamInitializer.Initialized)
		{
			Log.Info("[Sentry.NET] Steam not initialized");
			return false;
		}
		try
		{
			if (SaveManager.Instance.SettingsSave.FullConsole)
			{
				Log.Info("[Sentry.NET] Full console is on");
				return false;
			}
		}
		catch
		{
			Log.Info("[Sentry.NET] Exception while checking UploadData or FullConsole");
			return false;
		}
		if (ModManager.IsRunningModded())
		{
			Log.Info("[Sentry.NET] Is running modded");
			return false;
		}
		if (LocManager.Instance.OverridesActive)
		{
			Log.Info("[Sentry.NET] Loc overrides are active");
			return false;
		}
		if (ModManager.HasHarmonyPatches())
		{
			Log.Info("[Sentry.NET] Harmony patches active");
			return false;
		}
		return true;
	}
}
