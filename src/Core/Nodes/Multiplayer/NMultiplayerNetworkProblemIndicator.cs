using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Multiplayer.Quality;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Vfx;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Multiplayer;

public partial class NMultiplayerNetworkProblemIndicator : TextureRect
{
	private const float _qualityScoreToShowAt = 350f;

	private ulong _peerId;

	private Tween? _tween;

	public bool IsShown { get; private set; }

	public void Initialize(ulong peerId)
	{
		if (RunManager.Instance.NetService.Type.IsMultiplayer())
		{
			_peerId = peerId;
			TaskHelper.RunSafely(UpdateLoop());
		}
	}

	private async Task UpdateLoop()
	{
		while (RunManager.Instance.NetService.IsConnected && this.IsValid())
		{
			ConnectionStats statsForPeer = RunManager.Instance.NetService.GetStatsForPeer(_peerId);
			if (statsForPeer == null)
			{
				break;
			}
			float num = statsForPeer.PingMsec / (1f - statsForPeer.PacketLoss);
			bool flag = num >= 350f;
			if (!IsShown && flag)
			{
				_tween?.Kill();
				base.Visible = true;
				base.Modulate = Colors.White;
			}
			else if (IsShown && flag)
			{
				NUiFlashVfx nUiFlashVfx = NUiFlashVfx.Create(base.Texture, Colors.White);
				this.AddChildSafely(nUiFlashVfx);
				TaskHelper.RunSafely(nUiFlashVfx.StartVfx());
			}
			else if (IsShown && !flag)
			{
				_tween?.Kill();
				_tween = CreateTween();
				_tween.TweenProperty(this, "modulate:a", 0f, 0.5);
			}
			IsShown = flag;
			await Task.Delay(2000);
		}
	}
}
