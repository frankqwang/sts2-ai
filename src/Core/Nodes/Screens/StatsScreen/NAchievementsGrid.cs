using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.Achievements;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Platform;

namespace MegaCrit.Sts2.Core.Nodes.Screens.StatsScreen;

public partial class NAchievementsGrid : Control
{
	private Control _achievementsContainer;

	private bool _scrollbarPressed;

	private Vector2 _startDragPos;

	private Vector2 _targetDragPos;

	private bool _isDragging;

	public static IEnumerable<string> AssetPaths => from a in Enum.GetValues<Achievement>()
		select NAchievementHolder.GetPathForAchievement(a);

	private float ScrollLimitBottom => 0f - _achievementsContainer.Size.Y + base.Size.Y;

	public Control DefaultFocusedControl => _achievementsContainer.GetChildren().OfType<NAchievementHolder>().First();

	public override void _Ready()
	{
		_achievementsContainer = GetNode<Control>("%AchievementsContainer");
		List<NAchievementHolder> list = new List<NAchievementHolder>();
		Achievement[] values = Enum.GetValues<Achievement>();
		foreach (Achievement achievement in values)
		{
			NAchievementHolder nAchievementHolder = NAchievementHolder.Create(achievement);
			if (nAchievementHolder.IsUnlocked)
			{
				_achievementsContainer.AddChildSafely(nAchievementHolder);
			}
			else
			{
				list.Add(nAchievementHolder);
			}
		}
		foreach (NAchievementHolder item in list)
		{
			_achievementsContainer.AddChildSafely(item);
		}
	}

	private void OnAchievementsChanged()
	{
		foreach (NAchievementHolder item in _achievementsContainer.GetChildren().OfType<NAchievementHolder>())
		{
			item.RefreshUnlocked();
		}
	}

	public override void _EnterTree()
	{
		AchievementsUtil.AchievementsChanged += OnAchievementsChanged;
	}

	public override void _ExitTree()
	{
		AchievementsUtil.AchievementsChanged -= OnAchievementsChanged;
	}
}
