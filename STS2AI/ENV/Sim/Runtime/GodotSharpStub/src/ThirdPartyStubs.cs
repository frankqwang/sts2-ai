// Third-party library stubs for HeadlessSim
// These types are from external packages not included in the headless build.

#nullable enable
#pragma warning disable CS0414, CS0649, CS0108

// ── Steamworks ──
namespace Steamworks
{
    public static class SteamAPI { public static bool Init() => false; public static void Shutdown() { } }
    public static class SteamClient { public static void RunCallbacks() { } }
    public partial struct CSteamID { public ulong m_SteamID; public static CSteamID Nil => default; }
    public struct AppId_t { public uint m_AppId; }
    public enum EResult { k_EResultOK = 1 }
    public enum EChatRoomEnterResponse { k_EChatRoomEnterResponseSuccess = 1, k_EChatRoomEnterResponseDoesntExist, k_EChatRoomEnterResponseNotAllowed, k_EChatRoomEnterResponseFull, k_EChatRoomEnterResponseError, k_EChatRoomEnterResponseBanned, k_EChatRoomEnterResponseLimited, k_EChatRoomEnterResponseClanDisabled, k_EChatRoomEnterResponseCommunityBan, k_EChatRoomEnterResponseMemberBlockedYou, k_EChatRoomEnterResponseYouBlockedMember, k_EChatRoomEnterResponseRatelimitExceeded }
    public enum ESteamInputType { k_ESteamInputType_Unknown }
    public struct InputHandle_t { public ulong m_InputHandle; }
    public struct InputActionSetHandle_t { public ulong m_InputActionSetHandle; }
    public struct InputDigitalActionHandle_t { public ulong m_InputDigitalActionHandle; }
    public struct InputAnalogActionHandle_t { public ulong m_InputAnalogActionHandle; }
    public enum EInputActionOrigin { k_EInputActionOrigin_None }
    public enum ELobbyType { k_ELobbyTypePrivate, k_ELobbyTypeFriendsOnly, k_ELobbyTypePublic, k_ELobbyTypeInvisible }
    public enum ESteamNetworkingConfigDataType { k_ESteamNetworkingConfig_Int32, k_ESteamNetworkingConfig_Int64, k_ESteamNetworkingConfig_Float, k_ESteamNetworkingConfig_String, k_ESteamNetworkingConfig_Ptr }
    public enum ESteamNetworkingConfigValue { k_ESteamNetworkingConfig_TimeoutInitial = 24, k_ESteamNetworkingConfig_TimeoutConnected = 25 }
    public enum ESteamNetworkingConnectionState { k_ESteamNetworkingConnectionState_None, k_ESteamNetworkingConnectionState_Connecting, k_ESteamNetworkingConnectionState_FindingRoute, k_ESteamNetworkingConnectionState_Connected, k_ESteamNetworkingConnectionState_ClosedByPeer, k_ESteamNetworkingConnectionState_ProblemDetectedLocally }
    public enum EItemState { k_EItemStateNone }
    public struct CGameID { public ulong m_GameID; }
    public struct LobbyCreated_t { public EResult m_eResult; public ulong m_ulSteamIDLobby; }
    public struct LobbyEnter_t { public ulong m_ulSteamIDLobby; public uint m_EChatRoomEnterResponse; }
    public struct SteamNetworkingConfigValue_t { public ESteamNetworkingConfigValue m_eValue; public ESteamNetworkingConfigDataType m_eDataType; public long m_val; }
    public struct SteamNetworkingMessage_t { public System.IntPtr m_pData; public int m_cbSize; public uint m_conn; public long m_identityPeer; public void Release() { } }
    public struct SteamNetConnectionStatusChangedCallback_t { public uint m_hConn; public SteamNetConnectionInfo_t m_info; public ESteamNetworkingConnectionState m_eOldState; }
    public struct SteamNetConnectionInfo_t { public ESteamNetworkingConnectionState m_eState; public int m_eEndReason; public string? m_szEndDebug; public SteamNetworkingIdentity m_identityRemote; }
    public partial struct SteamNetworkingIdentity { public void SetSteamID(CSteamID id) { } public void SetSteamID64(ulong id) { } public ulong GetSteamID64() => 0; public CSteamID GetSteamID() => default; }
    public struct FriendGameInfo_t { public CGameID m_gameID; }
    public struct HSteamNetConnection { public uint m_HSteamNetConnection; }
    public struct HSteamListenSocket { public uint m_HSteamListenSocket; }
    public struct PublishedFileId_t { public ulong m_PublishedFileId; }
    public struct SteamAPICall_t { public ulong m_SteamAPICall; }
    public struct ItemInstalled_t { public PublishedFileId_t m_nPublishedFileId; public AppId_t m_unAppID; }
    public class Callback<T> where T : struct
    {
        public Callback(System.Action<T> func) { }
        public static Callback<T> Create(System.Action<T> func) => new(func);
        public void Dispose() { }
    }
    public class CallResult<T> where T : struct { public CallResult(System.Action<T, bool> func) { } public void Set(SteamAPICall_t call) { } public void Dispose() { } }
    public static partial class SteamFriends { public static bool GetFriendGamePlayed(CSteamID id, out FriendGameInfo_t info) { info = default; return false; } public static string GetFriendPersonaName(CSteamID id) => ""; }
    public static partial class SteamMatchmaking { public static SteamAPICall_t CreateLobby(ELobbyType type, int max) => default; public static CSteamID GetLobbyMemberByIndex(CSteamID id, int i) => default; public static CSteamID GetLobbyOwner(CSteamID id) => default; public static int GetNumLobbyMembers(CSteamID id) => 0; public static SteamAPICall_t JoinLobby(CSteamID id) => default; public static void LeaveLobby(CSteamID id) { } public static bool SetLobbyType(CSteamID id, ELobbyType type) => false; }
    public static partial class SteamNetworkingSockets { public static EResult AcceptConnection(uint conn) => EResult.k_EResultOK; public static bool CloseConnection(uint conn, int reason, string? debug, bool linger) => true; public static bool CloseListenSocket(uint socket) => true; public static uint ConnectP2P(ref SteamNetworkingIdentity id, int port, int n, SteamNetworkingConfigValue_t[] opts) => 0; public static uint CreateListenSocketP2P(int port, int n, SteamNetworkingConfigValue_t[] opts) => 0; public static int ReceiveMessagesOnConnection(uint conn, System.IntPtr[] msgs, int max) => 0; public static EResult SendMessageToConnection(uint conn, System.IntPtr data, uint size, int flags, out long num) { num = 0; return EResult.k_EResultOK; } }
    public static partial class SteamRemoteStorage { public static bool IsCloudEnabledForAccount() => false; public static bool IsCloudEnabledForApp() => false; }
    public static partial class SteamUGC { public static bool GetItemInstallInfo(PublishedFileId_t id, out ulong size, out string folder, uint folderSize, out uint ts) { size = 0; folder = ""; ts = 0; return false; } public static uint GetNumSubscribedItems() => 0; public static uint GetSubscribedItems(PublishedFileId_t[] items, uint max) => 0; public static uint GetItemState(PublishedFileId_t id) => 0; }
    public static partial class SteamUser { public static CSteamID GetSteamID() => default; }
    public static partial class SteamUtils { public static bool IsSteamRunningOnSteamDeck() => false; }
}

// ── HarmonyLib ──
namespace HarmonyLib
{
    public class Harmony { public Harmony(string id) { } public void PatchAll(System.Reflection.Assembly? assembly = null) { } }
    [System.AttributeUsage(System.AttributeTargets.Method)] public class HarmonyPatch : System.Attribute { public HarmonyPatch() { } public HarmonyPatch(System.Type type) { } public HarmonyPatch(System.Type type, string method) { } }
    [System.AttributeUsage(System.AttributeTargets.Method)] public class HarmonyPostfix : System.Attribute { }
    [System.AttributeUsage(System.AttributeTargets.Method)] public class HarmonyPrefix : System.Attribute { }
}

// ── Vortice DXGI ──
namespace Vortice.DXGI
{
    public class IDXGIFactory1 : System.IDisposable { public void Dispose() { } }
    public class IDXGIAdapter1 : System.IDisposable { public void Dispose() { } public AdapterDescription1 Description1; }
    public struct AdapterDescription1 { public string Description; public nuint DedicatedVideoMemory; }
    public static class DXGI { public static System.IDisposable? CreateDXGIFactory1<T>() => null; }
}
namespace SharpGen.Runtime { public class Result { public bool Success => true; } }

// ── MegaCrit Source Generation ──
namespace MegaCrit.Sts2.SourceGeneration
{
    [System.AttributeUsage(System.AttributeTargets.Interface | System.AttributeTargets.Class)]
    public class GenerateSubtypesAttribute : System.Attribute
    {
        public System.Diagnostics.CodeAnalysis.DynamicallyAccessedMemberTypes DynamicallyAccessedMemberTypes { get; set; }
    }
}

// ── MegaCrit addons ──
namespace MegaCrit.Sts2.addons.mega_text
{
    public partial class MegaTextLabel : Godot.Control { public string Text { get; set; } = ""; }
}
