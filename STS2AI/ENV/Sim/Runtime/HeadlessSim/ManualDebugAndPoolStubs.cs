using Godot;

namespace MegaCrit.Sts2.Core.Debug
{
    public enum BreadcrumbLevel
    {
        Debug,
        Info,
        Warning,
        Error,
        Fatal,
    }

    public enum SentryLevel
    {
        Debug,
        Info,
        Warning,
        Error,
        Fatal,
    }
}

namespace MegaCrit.Sts2.Core.Nodes.Pooling
{
    public partial class NodePool<T> : INodePool where T : Node
    {
        public NodePool(string scenePath, int prewarmCount) { }

        public IPoolable Get() => default!;

        public void Free(IPoolable poolable) { }
    }
}
