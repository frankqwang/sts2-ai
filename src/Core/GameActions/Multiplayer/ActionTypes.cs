using System;
using System.Collections.Generic;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;

namespace MegaCrit.Sts2.Core.GameActions.Multiplayer;

public static class ActionTypes
{
	private static readonly NetTypeCache<INetAction> _cache = new NetTypeCache<INetAction>(new List<Type>([..INetActionSubtypes.All, ..ReflectionHelper.GetSubtypesInMods<INetAction>()]));

	public static int TypeToId<T>() where T : INetAction
	{
		return _cache.TypeToId<T>();
	}

	private static int TypeToId(Type type)
	{
		return _cache.TypeToId(type);
	}

	public static int ToId(this INetAction message)
	{
		return _cache.TypeToId(message.GetType());
	}

	public static bool TryGetActionType(int id, out Type? type)
	{
		return _cache.TryGetTypeFromId(id, out type);
	}
}
