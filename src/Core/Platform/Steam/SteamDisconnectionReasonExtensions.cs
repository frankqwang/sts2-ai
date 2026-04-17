using System;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Logging;

namespace MegaCrit.Sts2.Core.Platform.Steam;

public static class SteamDisconnectionReasonExtensions
{
	public static SteamDisconnectionReason ToSteam(this NetError reason)
	{
		return (SteamDisconnectionReason)(1000 + reason);
	}

	public static NetError ToApp(this SteamDisconnectionReason steamReason)
	{
		if (steamReason >= SteamDisconnectionReason.AppGeneric && steamReason <= (SteamDisconnectionReason)1999)
		{
			NetError netError = (NetError)(steamReason - 1000);
			if (!Enum.IsDefined(netError))
			{
				Log.Error($"Received unknown application error from Steam: {steamReason}");
				return NetError.UnknownNetworkError;
			}
			return netError;
		}
		if (steamReason >= SteamDisconnectionReason.LocalMin)
		{
			if (steamReason <= SteamDisconnectionReason.LocalMax)
			{
				goto IL_0093;
			}
			if (steamReason <= SteamDisconnectionReason.MiscTimeout)
			{
				if (steamReason == SteamDisconnectionReason.RemoteTimeout || steamReason == SteamDisconnectionReason.MiscTimeout)
				{
					return NetError.Timeout;
				}
			}
			else
			{
				if ((uint)(steamReason - 5004) <= 1u)
				{
					goto IL_0093;
				}
				if (steamReason == SteamDisconnectionReason.RelayReceivedUnexpectedNoConnectionPacket)
				{
					return NetError.TryAgainLater;
				}
			}
		}
		else if (steamReason == SteamDisconnectionReason.None)
		{
			return NetError.None;
		}
		return NetError.UnknownNetworkError;
		IL_0093:
		return NetError.NoInternet;
	}
}
