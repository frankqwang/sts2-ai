using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using Sentry;

namespace MegaCrit.Sts2.Core.Debug;

public static class SentryService
{
	private const string _dsnSettingPath = "sentry/config/dsn";

	private static readonly StringName _sentrySdkSingleton = new StringName("SentrySDK");

	private static readonly StringName _sentryUserClass = new StringName("SentryUser");

	private static readonly StringName _idProperty = new StringName("id");

	private static readonly StringName _setUserMethod = new StringName("set_user");

	private static readonly StringName _setTagMethod = new StringName("set_tag");

	private static IDisposable? _sentryInstance;

	private static float _sampleRate = 1f;

	private static readonly string _sessionId = Guid.NewGuid().ToString();

	public static bool IsEnabled { get; private set; }

	public static string SessionId => _sessionId;

	public static void Initialize()
	{
		if (OS.HasFeature("editor") && !DisplayServer.GetName().Equals("headless", StringComparison.OrdinalIgnoreCase))
		{
			Log.Info("[Sentry.NET] Disabled in editor");
			return;
		}
		string dsn = GetDsn();
		if (string.IsNullOrEmpty(dsn))
		{
			Log.Info("[Sentry.NET] Disabled: no DSN configured in project settings");
			return;
		}
		ReleaseInfo releaseInfo = ReleaseInfoManager.Instance.ReleaseInfo;
		string environment = ((releaseInfo != null) ? "playtesters" : "development");
		string release = releaseInfo?.Version ?? "dev";
		_sentryInstance = SentrySdk.Init(delegate(SentryOptions options)
		{
			options.Dsn = dsn;
			options.Environment = environment;
			options.Release = release;
			options.Debug = false;
			options.AutoSessionTracking = true;
			options.IsGlobalModeEnabled = true;
			options.SendDefaultPii = false;
			options.SetBeforeSend(delegate(SentryEvent sentryEvent, SentryHint hint)
			{
				if (sentryEvent.Exception is AutoSlayTimeoutException)
				{
					return (SentryEvent?)null;
				}
				if (!HasUserConsent())
				{
					return (SentryEvent?)null;
				}
				if (ModManager.LoadedMods.Count > 0)
				{
					return (SentryEvent?)null;
				}
				return (System.Random.Shared.NextDouble() >= (double)_sampleRate) ? null : sentryEvent;
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
			if (releaseInfo != null)
			{
				scope.SetTag("branch", releaseInfo.Branch);
				scope.SetExtra("commit", releaseInfo.Commit);
				scope.SetExtra("build_date", releaseInfo.Date.ToString("o"));
			}
		});
		SetGdExtensionTag("session_id", _sessionId);
		Log.LogCallback += OnLogCallback;
		Log.Info("[Sentry.NET] Initialized: env=" + environment + ", release=" + release);
	}

	private static void OnLogCallback(LogLevel level, string message, int skipFrames)
	{
		if (IsEnabled)
		{
			switch (level)
			{
			case LogLevel.Error:
				SentrySdk.AddBreadcrumb(message, "log", null, null, BreadcrumbLevel.Error);
				break;
			case LogLevel.Warn:
				SentrySdk.AddBreadcrumb(message, "log", null, null, BreadcrumbLevel.Warning);
				break;
			}
		}
	}

	public static void SetUserContext(string uniqueId)
	{
		if (IsEnabled)
		{
			SentrySdk.ConfigureScope(delegate(Scope scope)
			{
				scope.User = new SentryUser
				{
					Id = uniqueId
				};
			});
			SetGdExtensionUser(uniqueId);
			Log.Debug("[Sentry.NET] User context set");
		}
	}

	private static void SetGdExtensionUser(string uniqueId)
	{
		try
		{
			if (Engine.HasSingleton(_sentrySdkSingleton))
			{
				GodotObject singleton = Engine.GetSingleton(_sentrySdkSingleton);
				GodotObject godotObject = ClassDB.Instantiate(_sentryUserClass).AsGodotObject();
				godotObject.Set(_idProperty, uniqueId);
				singleton.Call(_setUserMethod, godotObject);
			}
		}
		catch (Exception ex)
		{
			Log.Warn("[Sentry] Failed to set GDExtension user: " + ex.Message);
		}
	}

	private static void SetGdExtensionTag(string key, string value)
	{
		try
		{
			if (Engine.HasSingleton(_sentrySdkSingleton))
			{
				GodotObject singleton = Engine.GetSingleton(_sentrySdkSingleton);
				singleton.Call(_setTagMethod, key, value);
			}
		}
		catch (Exception ex)
		{
			Log.Warn("[Sentry] Failed to set GDExtension tag: " + ex.Message);
		}
	}

	public static void SetPlatformBranch(string? branch)
	{
		if (!IsEnabled)
		{
			return;
		}
		_sampleRate = branch switch
		{
			"public" => 0.1f, 
			"private-beta" => 1f, 
			"public-beta" => 0.2f, 
			_ => (branch != null) ? 0.1f : 0f, 
		};
		if (_sampleRate == 0f)
		{
			Log.Info("[Sentry.NET] Disabled: no platform branch (non-Steam build)");
			Shutdown();
			return;
		}
		SentrySdk.ConfigureScope(delegate(Scope scope)
		{
			scope.SetTag("platform.branch", branch);
		});
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

	public static void CaptureMessage(string message, SentryLevel level = SentryLevel.Info)
	{
		if (IsEnabled)
		{
			SentrySdk.CaptureMessage(message, level);
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
				scope.SetExtra("game.room_type", runState.CurrentRoom?.GetType().Name);
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

	private static bool HasUserConsent()
	{
		try
		{
			return SaveManager.Instance.PrefsSave.UploadData;
		}
		catch
		{
			return false;
		}
	}
}
