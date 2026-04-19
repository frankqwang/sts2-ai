namespace MegaCrit.Sts2.Core.Bindings.MegaSpine;

public readonly struct SpineAnimationAccess(MegaSprite? sprite)
{
	private readonly MegaSprite? _sprite = sprite;

	public bool IsValid => _sprite != null;

	public MegaTrackEntry? SetAnimation(string name, bool loop = true, int track = 0)
	{
		return _sprite?.GetAnimationState().SetAnimation(name, loop, track);
	}

	public MegaTrackEntry? AddAnimation(string name, float delay = 0f, bool loop = true, int track = 0)
	{
		return _sprite?.GetAnimationState().AddAnimation(name, delay, loop, track);
	}

	public MegaTrackEntry? GetCurrentTrack(int track = 0)
	{
		return _sprite?.GetAnimationState().GetCurrent(track);
	}

	public void SetTimeScale(float scale)
	{
		_sprite?.GetAnimationState().SetTimeScale(scale);
	}

	public MegaAnimationState? GetAnimationState()
	{
		return _sprite?.GetAnimationState();
	}
}
