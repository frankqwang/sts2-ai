global using SentryLevel = MegaCrit.Sts2.Core.Debug.SentryLevel;

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

namespace MegaCrit.Sts2.Core.Models.Monsters
{
    public static class DoormakerHeadlessExtensions
    {
        public static System.Threading.Tasks.Task AnimIn(this Doormaker doormaker) => System.Threading.Tasks.Task.CompletedTask;
    }
}

namespace MegaCrit.Sts2.Core.Nodes.Screens.DailyRun
{
    public partial struct DecodedDailyScore
    {
        public int victory;
        public int floors;
        public int badges;
        public int runTime;
        public bool isValid;
    }
}
