using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Godot.Collections;

/// <summary>Godot Dictionary stub — wraps System.Collections.Generic.Dictionary.</summary>
public class Dictionary : IDictionary<Variant, Variant>, IEnumerable<KeyValuePair<Variant, Variant>>
{
    private readonly System.Collections.Generic.Dictionary<string, Variant> _dict = new();

    public Dictionary() { }

    public Variant this[Variant key]
    {
        get => _dict.TryGetValue(key.ToString(), out var v) ? v : default;
        set => _dict[key.ToString()] = value;
    }

    public ICollection<Variant> Keys => _dict.Keys.Select(k => (Variant)k).ToList();
    public ICollection<Variant> Values => _dict.Values.ToList();
    public int Count => _dict.Count;
    public bool IsReadOnly => false;

    public void Add(Variant key, Variant value) => _dict[key.ToString()] = value;
    public bool ContainsKey(Variant key) => _dict.ContainsKey(key.ToString());
    public bool Remove(Variant key) => _dict.Remove(key.ToString());
    public bool TryGetValue(Variant key, out Variant value) => _dict.TryGetValue(key.ToString(), out value);
    public void Add(KeyValuePair<Variant, Variant> item) => _dict[item.Key.ToString()] = item.Value;
    public void Clear() => _dict.Clear();
    public bool Contains(KeyValuePair<Variant, Variant> item) => _dict.ContainsKey(item.Key.ToString());
    public void CopyTo(KeyValuePair<Variant, Variant>[] array, int arrayIndex) { }
    public bool Remove(KeyValuePair<Variant, Variant> item) => _dict.Remove(item.Key.ToString());
    public IEnumerator<KeyValuePair<Variant, Variant>> GetEnumerator() =>
        _dict.Select(kv => new KeyValuePair<Variant, Variant>((Variant)kv.Key, kv.Value)).GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}

/// <summary>Godot Dictionary&lt;TKey, TValue&gt; stub.</summary>
public class Dictionary<TKey, TValue> : IDictionary<TKey, TValue> where TKey : notnull
{
    private readonly System.Collections.Generic.Dictionary<TKey, TValue> _dict = new();

    public Dictionary() { }
    public Dictionary(System.Collections.Generic.IDictionary<TKey, TValue> dict) { foreach (var kv in dict) _dict[kv.Key] = kv.Value; }

    public TValue this[TKey key]
    {
        get => _dict[key];
        set => _dict[key] = value;
    }

    public ICollection<TKey> Keys => _dict.Keys;
    public ICollection<TValue> Values => _dict.Values;
    public int Count => _dict.Count;
    public bool IsReadOnly => false;

    public void Add(TKey key, TValue value) => _dict.Add(key, value);
    public bool ContainsKey(TKey key) => _dict.ContainsKey(key);
    public bool Remove(TKey key) => _dict.Remove(key);
    public bool TryGetValue(TKey key, out TValue value) => _dict.TryGetValue(key, out value!);
    public void Add(KeyValuePair<TKey, TValue> item) => _dict.Add(item.Key, item.Value);
    public void Clear() => _dict.Clear();
    public bool Contains(KeyValuePair<TKey, TValue> item) => _dict.ContainsKey(item.Key);
    public void CopyTo(KeyValuePair<TKey, TValue>[] array, int arrayIndex) => ((ICollection<KeyValuePair<TKey, TValue>>)_dict).CopyTo(array, arrayIndex);
    public bool Remove(KeyValuePair<TKey, TValue> item) => _dict.Remove(item.Key);
    public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator() => _dict.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}

/// <summary>Godot Array stub — wraps System.Collections.Generic.List.</summary>
public class Array : IList<Variant>
{
    private readonly List<Variant> _list = new();

    public Array() { }

    public Variant this[int index]
    {
        get => _list[index];
        set => _list[index] = value;
    }

    public int Count => _list.Count;
    public bool IsReadOnly => false;
    public void Add(Variant item) => _list.Add(item);
    public void Clear() => _list.Clear();
    public bool Contains(Variant item) => _list.Contains(item);
    public void CopyTo(Variant[] array, int arrayIndex) => _list.CopyTo(array, arrayIndex);
    public int IndexOf(Variant item) => _list.IndexOf(item);
    public void Insert(int index, Variant item) => _list.Insert(index, item);
    public bool Remove(Variant item) => _list.Remove(item);
    public void RemoveAt(int index) => _list.RemoveAt(index);
    public IEnumerator<Variant> GetEnumerator() => _list.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}

/// <summary>Godot Array&lt;T&gt; stub.</summary>
public class Array<T> : IList<T>
{
    private readonly List<T> _list = new();

    public Array() { }
    public Array(IEnumerable<T> items) { _list.AddRange(items); }

    public T this[int index]
    {
        get => _list[index];
        set => _list[index] = value;
    }

    public int Count => _list.Count;
    public bool IsReadOnly => false;
    public void Add(T item) => _list.Add(item);
    public void Clear() => _list.Clear();
    public bool Contains(T item) => _list.Contains(item);
    public void CopyTo(T[] array, int arrayIndex) => _list.CopyTo(array, arrayIndex);
    public int IndexOf(T item) => _list.IndexOf(item);
    public void Insert(int index, T item) => _list.Insert(index, item);
    public bool Remove(T item) => _list.Remove(item);
    public void RemoveAt(int index) => _list.RemoveAt(index);
    public IEnumerator<T> GetEnumerator() => _list.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
